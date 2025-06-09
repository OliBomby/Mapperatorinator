import math
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Adam_mini(Optimizer):
    def __init__(
            self,
            named_parameters: Iterable[Tuple[str, nn.Parameter]],
            lr: Union[float, torch.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            *,
            model_sharding: bool = None,
            dim: int = 2048,
            n_heads: int = 32,
            n_kv_heads: Optional[int] = None,
            verbose=True,
    ):

        '''
        This is the official implementation of Adam-mini (version 1.0.4).

        Paper: [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793).

        Github repo: https://github.com/zyushun/Adam-mini

        (Documentation omitted for brevity)
        '''

        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads

        # Force single GPU operation
        self.world_size = 1
        self.distributed = False

        if verbose and not self.distributed:
            print("Running on a single GPU. No distributed operations will be performed.")

        # Validation Checks
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not isinstance(self.dim, int):
            raise ValueError("Invalid dim value: {}".format(self.dim))
        if not isinstance(self.n_heads, int):
            raise ValueError("Invalid n_heads value: {}".format(self.n_heads))
        if not isinstance(self.n_kv_heads, int):
            raise ValueError("Invalid n_kv_heads value: {}".format(self.n_kv_heads))

        if model_sharding is not None and verbose:
            print(
                "UserWarning: model_sharding is deprecated since version 1.0.2. "
                "This argument is always set True. We will remove this argument in the future version.")

        # Parameter Groups Definitions
        self.embd_names = {"embed", "embd", "wte"}
        self.output_names = {"lm_head.weight", "output.weight"}
        self.wqk_names = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}
        self.mlp_names = {}

        optim_groups = []
        count_embd = count_output = count_wqk = 0
        for param_name, param in named_parameters:
            if not param.requires_grad:
                continue
            if verbose:
                print('Adam-mini found the param block with name:', param_name)
            state = {}
            state["name"] = param_name
            state["params"] = param
            zero_decay_param_names = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]
            if any(nd in param_name for nd in zero_decay_param_names):
                print(f"Adam-mini: weight decay set to 0.0 for {param_name}")
                state["weight_decay"] = 0.0
            else:
                state["weight_decay"] = weight_decay
            if any(embd_name in param_name for embd_name in self.embd_names):
                count_embd += 1
            if any(output_name in param_name for output_name in self.output_names):
                count_output += 1
            if any(wqk_name in param_name for wqk_name in self.wqk_names):
                count_wqk += 1
                assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
                state["head_numel"] = self.dim * self.dim // self.n_heads

            optim_groups.append(state)
        if verbose:
            print(
                f'Adam-mini found {count_embd} embedding layers, '
                f'{count_output} output layers, {count_wqk} Querys and Keys.')

        # Warnings
        if count_embd == 0 and verbose:
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, "
                "please check the name of your embedding layer and manually add them to 'self.embd_names' of Adam-mini. "
                "You can do this by adding an additional line of code: "
                "optimizer.embd_names.add('the name of your embedding layer'). ")
        if count_output == 0 and verbose:
            print(
                "=====>>> Warning by Adam-mini: No output layer found. If you are training Transformers (without weight-tying), "
                "please check the name of your output layer and manually add them to 'self.output_names' of Adam-mini. "
                "You can do this by adding an additional line of code: "
                "optimizer.output_names.add('the name of your output layer'). "
                "Please ignore this warning if you are using weight-tying.")
        if count_wqk == 0 and verbose:
            print(
                "=====>>>  Warning by Adam-mini: No Query or Key found. If you are training Transformers, "
                "please check the name of your Query and Key in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. "
                "You can do this by adding two additional lines of code: "
                "optimizer.wqk_names.add('the name of your Query' ); "
                "optimizer.wqk_names.add('the name of your Key'). ")

        if (count_output + count_embd + count_wqk == 0) and verbose:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. "
                "It can cause training instability on large-scale Transformers.")

        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        super().__init__(optim_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for p in group["params"]:

                state = self.state[p]
                if any(embd_name in name for embd_name in self.embd_names) or any(output_name in name for output_name in
                                                                                  self.output_names):  # this is for embedding and output layer
                    if p.grad is None:
                        continue
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["v"] = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)

                    grad = p.grad
                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = lr / bias_correction_1
                    p.addcdiv_(state["m"], h, value=-stepsize)

                elif any(wqk_name in name for wqk_name in self.wqk_names):  # this is for query and key
                    if p.grad is None:
                        continue
                    head_numel = group["head_numel"]
                    if len(state) == 0:
                        m = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)
                        state["m"] = m.view(-1, head_numel)
                        state["head_per_gpu"] = state["m"].size(0)  # this is head per gpu
                        state["step"] = 0
                        state["vmean"] = torch.zeros_like(
                            state["m"][0:state["head_per_gpu"], 0:1],
                            device=device,
                            memory_format=torch.preserve_format
                        )

                    grad = p.grad  # .to(torch.float32)
                    head_per_gpu = state["head_per_gpu"]
                    grad = grad.view(head_per_gpu, head_numel)
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(head_per_gpu, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)

                elif any(mlp_name in name for mlp_name in self.mlp_names):  # MLP blocks
                    if p.grad is None:
                        continue
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p.grad, device=device, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["neuron_per_gpu"] = state["m"].size(0)  # this is neuron per gpu
                        state["vmean"] = torch.zeros_like(
                            state["m"][0:state["neuron_per_gpu"], 0:1],
                            device=device,
                            memory_format=torch.preserve_format
                        )

                    grad = p.grad  # .to(torch.float32)
                    neuron_per_gpu = state["neuron_per_gpu"]
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(neuron_per_gpu, 1)
                    update = (state["m"] * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)

                else:  # other blocks
                    if len(state) == 0:
                        block_numel = torch.tensor(p.numel(), dtype=torch.float32, device=device)
                        state["m"] = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["reduced"] = False  # Since world_size is 1, no reduction
                        state["vmean"] = torch.zeros(1, device=device)
                        state["block_numel"] = block_numel.item()
                    if p.grad is None:
                        tmp_lr = torch.zeros(1, device=device)
                    else:
                        grad = p.grad  # .to(torch.float32)
                        tmp_lr = torch.sum(grad * grad)

                    # Since world_size is 1, skip all_reduce
                    if (self.world_size > 1):
                        # This block is skipped because world_size is set to 1
                        pass

                    if (p.grad is None):
                        continue
                    tmp_lr = tmp_lr / state["block_numel"]

                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1
                    state["m"].lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = state["m"] * stepsize
                    update.mul_(lr)
                    p.add_(-update)

        return loss
