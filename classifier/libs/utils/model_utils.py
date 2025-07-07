import os
from pathlib import Path

import lightning
import numpy as np
import torch
import torchmetrics
from omegaconf import DictConfig
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)
from torch.utils.data import DataLoader
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers.utils import cached_file

import routed_pickle

from ..dataset import OrsDataset, OsuParser
from ..model import OsuClassifier
from ..model.model import OsuClassifierOutput
from ..tokenizer import Tokenizer


class LitOsuClassifier(lightning.LightningModule):
    def __init__(self, args: DictConfig, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model: OsuClassifier = OsuClassifier(args, tokenizer)

    def forward(self, **kwargs) -> OsuClassifierOutput:
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        output: Seq2SeqSequenceClassifierOutput = self.model(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def testy_step(self, batch, batch_idx, prefix):
        output: Seq2SeqSequenceClassifierOutput = self.model(**batch)
        loss = output.loss
        preds = output.logits.argmax(dim=1)
        labels = batch["labels"]
        accuracy = torchmetrics.functional.accuracy(preds, labels, "multiclass", num_classes=self.args.data.num_classes)
        accuracy_10 = torchmetrics.functional.accuracy(output.logits, labels, "multiclass", num_classes=self.args.data.num_classes, top_k=10)
        accuracy_100 = torchmetrics.functional.accuracy(output.logits, labels, "multiclass", num_classes=self.args.data.num_classes, top_k=100)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_accuracy", accuracy)
        self.log(f"{prefix}_top_10_accuracy", accuracy_10)
        self.log(f"{prefix}_top_100_accuracy", accuracy_100)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.testy_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.testy_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.args)
        scheduler = get_scheduler(optimizer, self.args)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }}


def load_ckpt(ckpt_path, route_pickle=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(ckpt_path):
        ckpt_path = cached_file(ckpt_path, "model.ckpt")
    else:
        ckpt_path = Path(ckpt_path)

    checkpoint = torch.load(
        ckpt_path,
        map_location=lambda storage, loc: storage,
        weights_only=False,
        pickle_module=routed_pickle if route_pickle else None
    )
    tokenizer = checkpoint["hyper_parameters"]["tokenizer"]
    model_args = checkpoint["hyper_parameters"]["args"]
    state_dict = checkpoint["state_dict"]
    non_compiled_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model._orig_mod."):
            non_compiled_state_dict["model." + k[16:]] = v
        else:
            non_compiled_state_dict[k] = v

    model = LitOsuClassifier(model_args, tokenizer)
    model.load_state_dict(non_compiled_state_dict)
    model.eval().to(device)
    return model, model_args, tokenizer


def get_tokenizer(args: DictConfig) -> Tokenizer:
    return Tokenizer(args)


def get_optimizer(parameters, args: DictConfig) -> Optimizer:
    if args.optim.name == 'adamw':
        optimizer = AdamW(
            parameters,
            lr=args.optim.base_lr,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer: Optimizer, args: DictConfig, num_processes=1) -> LRScheduler:
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps * num_processes,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps * num_processes - args.optim.warmup_steps * num_processes,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps * num_processes],
    )

    return scheduler


def get_dataloaders(tokenizer: Tokenizer, args: DictConfig) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(args, tokenizer)
    dataset = {
        "train": OrsDataset(
            args.data,
            parser,
            tokenizer,
        ),
        "test": OrsDataset(
            args.data,
            parser,
            tokenizer,
            test=True,
        ),
    }

    dataloaders = {}
    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.dataloader.num_workers > 0,
            worker_init_fn=worker_init_fn,
        )

    return dataloaders["train"], dataloaders["test"]


def worker_init_fn(worker_id: int) -> None:
    """
    Give each dataloader a unique slice of the full dataset.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        np.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)
