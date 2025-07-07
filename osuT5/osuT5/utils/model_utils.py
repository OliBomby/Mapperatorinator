import multiprocessing
import time
from multiprocessing.managers import Namespace

import torch
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR, ConstantLR,
)

from ..dataset.ors_dataset import OrsDataset
from ..dataset.osu_parser import OsuParser
from ..dataset.mmrs_dataset import MmrsDataset
from ..event import EventType
from ..model.configuration_mapperatorinator import MapperatorinatorConfig
from ..model.modeling_mapperatorinator import Mapperatorinator
from ..tokenizer import Tokenizer
from ..config import TrainConfig


def get_shared_training_state() -> Namespace:
    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    shared.current_epoch = 1
    shared.last_log = time.time()
    shared.current_loss = np.inf
    shared.best_loss = np.inf
    return shared


def get_model_config(args: TrainConfig, tokenizer: Tokenizer) -> MapperatorinatorConfig:
    return MapperatorinatorConfig(
        backbone_model_name=args.model.name,
        backbone_overwrite=args.model.overwrite,
        backbone_add_config=args.model.add_config,
        flash_attention=args.flash_attention,
        vocab_size_in=tokenizer.vocab_size_in,
        vocab_size_out=tokenizer.vocab_size_out,
        num_classes=tokenizer.num_classes,
        num_mappers=tokenizer.num_mapper_classes,
        input_features=args.model.input_features,
        project_encoder_input=args.model.project_encoder_input,
        embed_decoder_input=args.model.embed_decoder_input,
        do_style_embed=args.model.do_style_embed,
        do_difficulty_embed=args.model.do_difficulty_embed,
        do_mapper_embed=args.model.do_mapper_embed,
        do_song_position_embed=args.model.do_song_position_embed,
        cond_dim=args.model.cond_dim,
        cond_size=args.model.cond_size,
        spectrogram_implementation=args.model.spectrogram.implementation,
        spectrogram_log_scale=args.model.spectrogram.log_scale,
        sample_rate=args.model.spectrogram.sample_rate,
        n_fft=args.model.spectrogram.n_fft,
        n_mels=args.model.spectrogram.n_mels,
        hop_length=args.model.spectrogram.hop_length,
        f_min=args.model.spectrogram.f_min,
        f_max=args.model.spectrogram.f_max,
        pad_mode=args.model.spectrogram.pad_mode,
        rhythm_weight=args.data.rhythm_weight,
        rhythm_token_start=tokenizer.event_start[EventType.TIME_SHIFT],
        rhythm_token_end=tokenizer.event_end[EventType.TIME_SHIFT],
        label_smoothing=args.data.label_smoothing,
        src_seq_len=args.data.src_seq_len,
        tgt_seq_len=args.data.tgt_seq_len,
        rope_type=args.model.rope_type,
        rope_encoder_scaling_factor=args.model.rope_encoder_scaling_factor,
        rope_decoder_scaling_factor=args.model.rope_decoder_scaling_factor,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.sos_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.sos_id,
        max_length=args.data.tgt_seq_len,
    )


def get_model(args: TrainConfig, tokenizer: Tokenizer) -> Mapperatorinator:
    model = Mapperatorinator(get_model_config(args, tokenizer))
    return model


def get_tokenizer(args: TrainConfig) -> Tokenizer:
    return Tokenizer(args)


def get_optimizer(model: Mapperatorinator, args: TrainConfig) -> Optimizer:
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

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

    if args.optim.name == 'adamw':
        from torch.optim import AdamW
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
        from torch.optim import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'muon':
        from .muon_utils import Muon
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

        optimizer = Muon(
            muon_params=muon_params,
            lr=args.optim.base_lr,
            adamw_lr=args.optim.base_lr_2,
            adamw_params=adamw_params,
            adamw_betas=(0.90, 0.95),
            adamw_wd=args.optim.weight_decay,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_scheduler(optimizer: Optimizer, args: TrainConfig, accelerator) -> LRScheduler:
    step = 0
    schedulers = []
    milestones = []

    if args.optim.warmup_steps > 0:
        schedulers.append(LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps * accelerator.num_processes,
        ))
        step += args.optim.warmup_steps * accelerator.num_processes
        milestones.append(step)

    if args.optim.sustain_steps > 0:
        schedulers.append(ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=args.optim.sustain_steps * accelerator.num_processes,
        ))
        step += args.optim.sustain_steps * accelerator.num_processes
        milestones.append(step)

    if args.optim.lr_scheduler == "cosine":
        schedulers.append(CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps * accelerator.num_processes - step,
            eta_min=args.optim.final_cosine,
        ))
    elif args.optim.lr_scheduler == "linear":
        schedulers.append(LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.optim.final_cosine / args.optim.base_lr,
            total_iters=args.optim.total_steps * accelerator.num_processes - step,
        ))

    scheduler = SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )

    return scheduler


def get_dataset(args: TrainConfig, test: bool, **kwargs) -> Dataset:
    if args.data.dataset_type == "ors":
        return OrsDataset(args=args.data, test=test, **kwargs)
    elif args.data.dataset_type == "mmrs":
        return MmrsDataset(args=args.data, **kwargs)
    else:
        raise NotImplementedError


def get_dataloaders(tokenizer: Tokenizer, args: TrainConfig, shared: Namespace) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(args, tokenizer)
    dataset = {
        "train": get_dataset(
            args=args,
            test=False,
            parser=parser,
            tokenizer=tokenizer,
            shared=shared,
        ),
        "test": get_dataset(
            args=args,
            test=True,
            parser=parser,
            tokenizer=tokenizer,
            shared=shared,
        ),
    }

    dataloaders = {}
    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        dataloaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=args.dataloader.num_workers,
            pin_memory=args.dataloader.pin_memory,
            drop_last=args.dataloader.drop_last,
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
