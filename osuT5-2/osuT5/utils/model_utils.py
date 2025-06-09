import os

import multiprocessing
import time
from multiprocessing.managers import Namespace
from .adam_mini import Adam_mini
import torch

from typing import Callable
import numpy as np
from omegaconf import DictConfig, open_dict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    ConstantLR,
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)

from ..dataset import OrsDataset, OsuParser
from ..model.osu_t import OsuT
from ..tokenizer import Tokenizer


def get_shared_training_state() -> Namespace:
    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    shared.current_epoch = 1
    shared.last_log = time.time()
    shared.current_loss = np.inf
    shared.best_loss = np.inf
    return shared


def get_model(args: DictConfig, tokenizer: Tokenizer) -> OsuT:
    model = OsuT(args, tokenizer)
    print(model)
    return model


def get_tokenizer(args: DictConfig) -> Tokenizer:
    return Tokenizer(args)


def get_optimizer(model: OsuT, args: DictConfig) -> Optimizer:
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]
    
    # selctive weight decay
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # total weight decay
    optimizer_grouped_parameters2 = [
        {
            "params": [ p for n, p in model.named_parameters()],
            "weight_decay": args.optim.weight_decay,
        },
        
    ]
    
    
    if args.optim.name == 'adamw':
        from transformers import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    elif args.optim.name == 'AdEMAMix':
        from .ademamix import AdEMAMix
        optimizer = AdEMAMix(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            weight_decay=args.optim.weight_decay,
            alpha=8.0,
            betas=(0.9, 0.999, 0.9999)
        )

    elif args.optim.name == 'soap':
        from .fsdp_optimizers.soap import SOAP
        optimizer = SOAP(
            model.parameters(),
            lr=args.optim.base_lr,
            weight_decay=args.optim.weight_decay,
        )

    elif args.optim.name == 'ForeachSOAP':
        from heavyball import ForeachSOAP
        optimizer = ForeachSOAP(
            optimizer_grouped_parameters2,
            lr=args.optim.base_lr,
            weight_decay=args.optim.weight_decay,
        )
    elif args.optim.name == 'mudamw':
         from .mudamw import AdamW

         muon_exclude = {param for name, param in model.named_parameters()
            if (any(kw in name.lower() for kw in {'embed', 'proj_out'}) or param.ndim <= 1)
         }
         optimizer = AdamW(
                model.parameters(),
                lr=args.optim.base_lr,
                weight_decay=args.optim.weight_decay,
                correct_bias=True,
                no_deprecation_warning=False,
                cautious=False,
                orthogonal_init=False,
                muon_exclude=muon_exclude,
            )

    elif args.optim.name == 'muon':
        """
        Muon is intended to optimize only the internal ≥2D parameters of a network. 
        Embeddings, classifier heads, and scalar or vector parameters should be optimized using AdamW.
        """
        adamw_params = [
            param for name, param in model.named_parameters()
            if (any(kw in name.lower() for kw in {'embed', 'proj_out'}) or param.ndim <= 1)
        ]
        
        adamw_param_set = set(adamw_params)
        muon_params = [
            param for _, param in model.named_parameters()
            if param not in adamw_param_set
        ]


        print(f"Number of parameters for Muon: {len(muon_params)}")
        print(f"Number of parameters for AdamW: {len(adamw_params)}")
        # print muon param names

        from .fsdp_optimizers.muon import Muon
        optimizer = Muon(muon_params=muon_params, lr=args.optim.base_lr, adamw_lr=args.optim.adamw_lr, adamw_params=adamw_params,
   adamw_betas=(0.90, 0.95), adamw_wd=args.optim.weight_decay)

        
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer: Optimizer, args: DictConfig, accelerator) -> LRScheduler:
    
    def get_lr(it):
        assert it <= args.optim.total_steps
        # 1) linear warmup for warmup_iters steps
        if it < args.optim.warmup_steps:
            return (it+1) / args.optim.warmup_steps
        # 2) constant lr for a while
        elif it < args.optim.total_steps - args.optim.warmdown_steps:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.optim.total_steps - it) / args.optim.warmdown_steps
            return decay_ratio
    if args.optim.lr_scheduler == "Triangular":
       scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
       return scheduler
    
    if args.optim.lr_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.75,
            patience=args.optim.patience,
            verbose=True,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )
        return scheduler
    scheduler_p1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps * accelerator.num_processes,
        last_epoch=-1,
    )

    scheduler_p2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps * accelerator.num_processes - args.optim.warmup_steps * accelerator.num_processes,
        eta_min=args.optim.final_cosine,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_p1, scheduler_p2],
        milestones=[args.optim.warmup_steps * accelerator.num_processes],
    )
    if args.optim.lr_scheduler == "constant":
        scheduler = ConstantLR(optimizer)

    return scheduler


def get_dataloaders(tokenizer: Tokenizer, args: DictConfig, shared: Namespace) -> tuple[Callable[[], DataLoader], DataLoader]:
    parser = OsuParser(args, tokenizer)
    dataset = {
        "train": OrsDataset(
            args.data,
            parser,
            tokenizer,
            beatmap_files = None,
            shared=shared,
        ),
        "test": OrsDataset(
            args.data,
            parser,
            tokenizer,
            beatmap_files = None,
            test=True,
            shared=shared,
        ),
    }


    batch_size = args.optim.batch_size // args.optim.grad_acc
    train_dataloader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers= True,
            timeout=30, # 10 seconds
            # worker_init_fn=worker_init_fn,
        )

    # Prepare test DataLoader once as it is used for evaluation
    batch_size = args.optim.batch_size // args.optim.grad_acc
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        num_workers=args.dataloader.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        #worker_init_fn=worker_init_fn,
    )

    return train_dataloader, test_dataloader


def worker_init_fn(worker_id: int) -> None:
    """
    Give each dataloader a unique slice of the full dataset.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    print(f"Worker {worker_id} processing data from {overall_start} to {overall_end}")
    # configure the dataset to only process the split workload
    # per_worker = int(
    #     np.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    # )
    # dataset.start = overall_start + worker_id * per_worker
    # dataset.end = min(dataset.start + per_worker, overall_end)
