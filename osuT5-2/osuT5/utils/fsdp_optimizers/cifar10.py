import logging
import torch
import torch.distributed
import torch.nn as nn
from einops import rearrange
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from tqdm import tqdm
from soap import SOAP
from kron import Kron
from muon import Muon
from kron_mars import KronMars

debug = False
optimizer = "kron_mars"
optimizers = {
    "kron": Kron,
    "soap": SOAP,
    "muon": Muon,
    "kron_mars": KronMars,
}
optimizer_class = optimizers[optimizer]

def print_if_master(*args):
    if torch.distributed.get_rank() == 0 and debug:
        print(*args)


torch.distributed.init_process_group(
    backend="nccl", init_method="env://",
)
torch.cuda.set_device(torch.distributed.get_rank())


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attentions = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            for _ in range(depth)
        ])
        self.ffns = nn.ModuleList([
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            for _ in range(depth)
        ])
    def forward(self, x):
        for i in range(len(self.attentions)):
            x = self.attentions[i](x) + x
            x = self.ffns[i](x) + x
        return x


class AttnPooler(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.q = torch.nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.q.expand(x.shape[0], -1, -1)
        out = F.scaled_dot_product_attention(q, k, v).squeeze(1)
        return out

class PatchEmbed(nn.Module):

    def __init__(
        self,
        channels=3,
        patch_size=16,
        out_dim=768,
        bias=True,
        pre_norm=False,
        post_norm=False,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]

        patch_dim = channels * patch_size[0] * patch_size[1]

        self.pre_norm = torch.nn.LayerNorm(patch_dim) if pre_norm else torch.nn.Identity()
        self.post_norm = torch.nn.LayerNorm(out_dim) if post_norm else torch.nn.Identity()
        self.proj = torch.nn.Linear(patch_dim, out_dim, bias=bias)

    def forward(self, x):
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_height, p2=self.patch_width)
        x = self.pre_norm(x)
        x = self.proj(x)
        x = self.post_norm(x)
        return x

class VIT(nn.Module):

    def __init__(
        self,
        num_layers=8,
        head_dim=64,
        heads=8,
        mlp_mult=4,
        dropout=0.1,
        emb_dropout=0.1,
        num_classes=10,
        in_channels=3,
        patch_size=4,
        image_size=32,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            channels=3, patch_size=patch_size, out_dim=head_dim * heads, bias=True, pre_norm=False, post_norm=True
        )
        dim = head_dim * heads
        self.pos_embs = torch.nn.Parameter(
            torch.randn((image_size // patch_size) ** 2, head_dim * heads)
        )

        self.transformer = Transformer(
            dim=dim,
            depth=num_layers,
            heads=heads,
            dim_head=head_dim,
            mlp_dim=dim * mlp_mult,
            dropout=dropout,
        )

        self.out_norm = torch.nn.LayerNorm(dim)
        self.pooler = AttnPooler(dim) # I had issues indexing the cls token, so we'll do pooling instead
        self.proj_out = torch.nn.Linear(dim, num_classes)

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values)
        h = x + self.pos_embs.unsqueeze(0)
        h = self.transformer(h)
        h = self.out_norm(h)
        h = self.pooler(h)
        h = self.proj_out(h)
        return h



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 512

# hack if you're downloading for first time
if torch.distributed.get_rank() == 0:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
torch.distributed.barrier()
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=8)

if torch.distributed.get_rank() == 0:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
torch.distributed.barrier()
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = VIT()

device_mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("dp",))

fsdp_config = {
    "mesh": device_mesh,
    "reshard_after_forward": True,
}

for attn in net.transformer.attentions:
    print_if_master("Sharding attention")
    fully_shard(attn, **fsdp_config)
for ffn in net.transformer.ffns:
    print_if_master("Sharding ffn")
    fully_shard(ffn, **fsdp_config)
print_if_master("Sharding transformer")
fully_shard(net, **fsdp_config)

criterion = nn.CrossEntropyLoss()
optimizer = optimizer_class(net.parameters(), lr=0.001)


pbar = tqdm(enumerate(trainloader), total=len(trainloader), disable=torch.distributed.get_rank() != 0)
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")

        # zero the parameter gradients
        optimizer.zero_grad()
        print_if_master("zeroed grad")

        # forward + backward + optimize
        outputs = net(inputs)
        print_if_master("got outputs")
        loss = criterion(outputs, labels)
        print_if_master("got loss")
        loss.backward()
        print_if_master("backprop'd")
        optimizer.step()
        print_if_master("stepped")

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.3f}")

print('Finished Training')
