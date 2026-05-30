import json
import multiprocessing
import re
import time
from multiprocessing.managers import Namespace
from pathlib import Path

import torch
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterableDataset, default_collate
from torch.optim.lr_scheduler import (
    LRScheduler,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR, ConstantLR,
)
from transformers.utils import cached_file

from ..dataset.osu_parser import OsuParser
from ..event import EventType
from ..model.configuration_mapperatorinator import MapperatorinatorConfig
from ..model.modeling_mapperatorinator import Mapperatorinator
from ..tokenizer import Tokenizer
from ..config import TrainConfig


LORA_METADATA_FILENAME = "mapperatorinator_lora_metadata.json"
_GAMEMODE_SUBFOLDER_PATTERN = re.compile(r"^gamemode=(\d+)$")


def get_shared_training_state() -> Namespace:
    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    shared.current_epoch = 1
    shared.last_log = time.time()
    shared.current_loss = np.inf
    shared.best_loss = np.inf
    return shared


def _get_model_config(
        args: TrainConfig,
        tokenizer: Tokenizer,
        dtype: torch.dtype,
        attn_implementation: str,
) -> MapperatorinatorConfig:
    return MapperatorinatorConfig(
        backbone_model_name=args.model.name,
        backbone_overwrite=args.model.overwrite,
        backbone_add_config=args.model.add_config,
        vocab_size_in=tokenizer.vocab_size_in,
        vocab_size_out=tokenizer.vocab_size_out,
        num_classes=tokenizer.num_classes,
        num_mappers=tokenizer.num_mapper_classes,
        input_features=args.model.input_features,
        input_raw_wave=args.model.input_raw_wave,
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
        rope_scaling=args.model.rope_scaling,
        deterministic_flash_attn=args.model.deterministic_flash_attn,
        attention_bias=args.model.attention_bias,
        global_attn_every_n_layers=args.model.global_attn_every_n_layers,
        local_attention=args.model.local_attention,
        local_rope_theta=args.model.local_rope_theta,
        global_rope_theta=args.model.global_rope_theta,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.sos_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.sos_id,
        max_length=args.data.tgt_seq_len,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def _get_model(
        args: TrainConfig,
        tokenizer: Tokenizer,
        dtype: torch.dtype,
        attn_implementation: str,
) -> Mapperatorinator:
    model = Mapperatorinator(_get_model_config(
        args,
        tokenizer,
        dtype,
        attn_implementation,
    ))
    return model


def _precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "amp":
        return torch.float32  # Handled separately with autocast
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def _normalize_ckpt_path(ckpt_path: str | Path | None) -> str | Path | None:
    if not ckpt_path:
        return None

    path = Path(ckpt_path)
    return path if path.exists() else str(ckpt_path)


def _is_local_custom_checkpoint(ckpt_path: str | Path | None) -> bool:
    return isinstance(ckpt_path, Path) and (ckpt_path / "pytorch_model.bin").exists() and (ckpt_path / "custom_checkpoint_0.pkl").exists()


def _normalize_ckpt_subfolder(ckpt_subfolder: str | None) -> str:
    if not ckpt_subfolder:
        return ""
    return ckpt_subfolder.strip().replace("\\", "/").strip("/")


def _normalize_ckpt_subfolders(ckpt_subfolders: list[str] | None) -> list[str] | None:
    if ckpt_subfolders is None:
        return None
    return sorted({_normalize_ckpt_subfolder(ckpt_subfolder) for ckpt_subfolder in ckpt_subfolders})


def get_lora_checkpoint_metadata(args: TrainConfig) -> dict:
    return {
        "format_version": 1,
        "ckpt_subfolders": _normalize_ckpt_subfolders(args.lora_metadata.ckpt_subfolders),
    }


def save_lora_checkpoint_metadata(output_dir: str | Path, args: TrainConfig) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / LORA_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(get_lora_checkpoint_metadata(args), f, indent=2, sort_keys=True)
        f.write("\n")
    return metadata_path


def load_lora_checkpoint_metadata(lora_path: str | Path | None) -> dict | None:
    lora_path = _normalize_ckpt_path(lora_path)
    if not lora_path:
        return None

    if isinstance(lora_path, Path):
        metadata_path = lora_path / LORA_METADATA_FILENAME
        if not metadata_path.is_file():
            return None
    else:
        try:
            metadata_path = cached_file(
                lora_path,
                LORA_METADATA_FILENAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        except Exception:
            metadata_path = None

        if metadata_path is None:
            return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Failed to read LoRA metadata from {metadata_path}: {exc}")
        return None

    ckpt_subfolders = metadata.get("ckpt_subfolders")
    if ckpt_subfolders is not None:
        if not isinstance(ckpt_subfolders, list) or not all(isinstance(item, str) for item in ckpt_subfolders):
            print(f"Warning: Invalid LoRA checkpoint subfolder metadata in {metadata_path}: {ckpt_subfolders}")
            return None
        metadata["ckpt_subfolders"] = _normalize_ckpt_subfolders(ckpt_subfolders)
    else:
        metadata["ckpt_subfolders"] = None

    return metadata


def get_model_checkpoint_subfolder(ckpt_path: str | Path | None, ckpt_subfolder: str | None = None) -> str:
    if ckpt_subfolder:
        return _normalize_ckpt_subfolder(ckpt_subfolder)

    if isinstance(ckpt_path, Path):
        if _GAMEMODE_SUBFOLDER_PATTERN.fullmatch(ckpt_path.name):
            return ckpt_path.name
        return ""

    if isinstance(ckpt_path, str):
        for part in ckpt_path.replace("\\", "/").split("/"):
            if _GAMEMODE_SUBFOLDER_PATTERN.fullmatch(part):
                return part

    return ""


def resolve_compatible_lora_path(
        lora_path: str | Path | None,
        *,
        ckpt_subfolder: str | None = None,
        verbose: bool = True,
) -> tuple[str | Path | None, dict | None]:
    lora_path = _normalize_ckpt_path(lora_path)
    if not lora_path:
        return None, None

    metadata = load_lora_checkpoint_metadata(lora_path)
    if metadata is None:
        return lora_path, None

    compatible_ckpt_subfolders = metadata.get("ckpt_subfolders")
    ckpt_subfolder = _normalize_ckpt_subfolder(ckpt_subfolder)
    if compatible_ckpt_subfolders is None:
        return lora_path, metadata

    if compatible_ckpt_subfolders is not None and ckpt_subfolder not in compatible_ckpt_subfolders:
        if verbose:
            print(
                f"Skipping LoRA {lora_path}: it supports checkpoint subfolders "
                f"{compatible_ckpt_subfolders}, not {repr(ckpt_subfolder)}."
            )
        return None, metadata

    return lora_path, metadata


def _format_model_source(ckpt_path: str | Path | None, subfolder: str | None = None) -> str:
    if not ckpt_path:
        return ""

    source = ckpt_path.as_posix() if isinstance(ckpt_path, Path) else str(ckpt_path)
    return f"{source}/{subfolder}" if subfolder else source


def resolve_model_checkpoint_path(
        ckpt_path: str | Path | None,
        gamemode: int | None = None,
        auto_select_gamemode_model: bool = True,
) -> tuple[str | Path | None, str | None]:
    ckpt_path = _normalize_ckpt_path(ckpt_path)
    if not ckpt_path or gamemode is None or not auto_select_gamemode_model:
        return ckpt_path, ""

    subfolder = f"gamemode={gamemode}"

    if isinstance(ckpt_path, Path):
        gamemode_path = ckpt_path / subfolder
        if gamemode_path.is_dir():
            return gamemode_path, None
        return ckpt_path, ""

    try:
        subdir_tokenizer = cached_file(
            ckpt_path,
            "tokenizer.json",
            subfolder=subfolder,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
    except Exception:
        subdir_tokenizer = None

    if subdir_tokenizer is not None:
        return ckpt_path, subfolder

    return ckpt_path, ""


def load_model(ckpt_path: str | Path | None, t5_args: TrainConfig, device, precision: str = "fp32", attn_implementation: str = "sdpa",
               eval_mode: bool = True, pickle_module=None, gamemode: int | None = None,
               auto_select_gamemode_model: bool = True):
    model_loader, tokenizer_loader = load_model_loaders(
        ckpt_path,
        t5_args,
        device,
        precision,
        attn_implementation,
        eval_mode,
        pickle_module,
        gamemode=gamemode,
        auto_select_gamemode_model=auto_select_gamemode_model,
    )
    return model_loader(), tokenizer_loader()


def load_model_loaders(
        ckpt_path: str | Path | None,
        t5_args: TrainConfig,
        device,
        precision: str = "fp32",
        attn_implementation: str = "sdpa",
        eval_mode: bool = True,
        pickle_module=None,
        lora_path=None,
        gamemode: int | None = None,
        auto_select_gamemode_model: bool = True,
):
    if not ckpt_path:
        if eval_mode:
            raise ValueError("Model path is empty.")
        else:
            print("No pretrained model path provided, training from scratch.")

    requested_ckpt_path = _normalize_ckpt_path(ckpt_path)
    ckpt_path, ckpt_subfolder = resolve_model_checkpoint_path(
        ckpt_path,
        gamemode=gamemode,
        auto_select_gamemode_model=auto_select_gamemode_model,
    )

    requested_source = _format_model_source(requested_ckpt_path)
    resolved_source = _format_model_source(ckpt_path, ckpt_subfolder)
    if requested_source and requested_source != resolved_source:
        print(f"Using gamemode-specific model checkpoint: {resolved_source}")

    lora_path, _ = resolve_compatible_lora_path(
        lora_path,
        ckpt_subfolder=get_model_checkpoint_subfolder(ckpt_path, ckpt_subfolder),
    )

    def tokenizer_loader():
        if not ckpt_path:
            tokenizer = get_tokenizer(t5_args)
        elif not _is_local_custom_checkpoint(ckpt_path):
            tokenizer = Tokenizer.from_pretrained(
                ckpt_path.as_posix() if isinstance(ckpt_path, Path) else ckpt_path,
                subfolder=ckpt_subfolder,
            )
        else:
            tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=pickle_module, weights_only=False)
            tokenizer = Tokenizer()
            tokenizer.load_state_dict(tokenizer_state)
        return tokenizer

    tokenizer = tokenizer_loader()

    def model_loader():
        dtype = _precision_to_dtype(precision)
        if not ckpt_path:
            model = _get_model(t5_args, tokenizer, dtype=dtype, attn_implementation=attn_implementation)
            model.to(device=device, dtype=dtype)
        elif not _is_local_custom_checkpoint(ckpt_path):
            model = Mapperatorinator.from_pretrained(
                ckpt_path.as_posix() if isinstance(ckpt_path, Path) else ckpt_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
                device_map=device,
                subfolder=ckpt_subfolder,
            )
            model.generation_config.disable_compile = True
        else:
            model_state = torch.load(ckpt_path / "pytorch_model.bin", weights_only=True)
            model = _get_model(t5_args, tokenizer, dtype=dtype, attn_implementation=attn_implementation)
            if t5_args.pretrained_t5_compat:
                del model_state["shared.weight"]
                del model_state["encoder.embed_tokens.weight"]
                del model_state["decoder.embed_tokens.weight"]
                del model_state["lm_head.weight"]
                model.transformer.load_state_dict(model_state, strict=False)
            else:
                model.load_state_dict(model_state)
            model.to(device=device, dtype=dtype)

        if lora_path is not None:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError("Please install the 'peft' library to use LoRA fine-tuning.")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print(f"Loaded LoRA weights from {lora_path}")

        if eval_mode:
            model.eval()

        print(f"Model loaded: {resolved_source} on device {device}")
        return model

    return model_loader, tokenizer_loader


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


def get_dataset(args: TrainConfig, **kwargs) -> IterableDataset:
    if args.data.dataset_type == "ors":
        from ..dataset.ors_dataset import OrsDataset
        return OrsDataset(args=args.data, **kwargs)
    elif args.data.dataset_type == "mmrs":
        from ..dataset.mmrs_dataset import MmrsDataset
        return MmrsDataset(args=args.data, **kwargs)
    elif args.data.dataset_type == "web":
        from ..dataset.web_dataset import WebDataset
        return WebDataset(args=args.data, **kwargs)
    else:
        raise NotImplementedError


def get_dataloaders(tokenizer: Tokenizer, args: TrainConfig, shared: Namespace) -> tuple[DataLoader, DataLoader]:
    parser = OsuParser(args, tokenizer)
    datasets = {
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
        dataset = datasets[split]
        batch_size = args.optim.batch_size // args.optim.grad_acc
        num_indices = args.data.train_dataset_end - args.data.train_dataset_start if split == "train" else args.data.test_dataset_end - args.data.test_dataset_start
        if num_indices < args.dataloader.num_workers:
            print(f"Warning: Number of {split} samples ({num_indices}) is less than the number of dataloader workers ({args.dataloader.num_workers}). Reducing num_workers to {num_indices}.")
            num_workers = num_indices
        else:
            num_workers = args.dataloader.num_workers

        dataloader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=args.dataloader.pin_memory,
            drop_last=args.dataloader.drop_last,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn if args.data.dataset_type in ["ors", "mmrs"] else None,
        )

        if args.dataloader.balancer_buffer_size > 0 and num_workers > 0:
            # Empty the whole balancer buffer into the prefetch buffer, so it starts filling the balancer buffer immediately while training
            dataloader_kwargs["prefetch_factor"] = int(args.dataloader.balancer_buffer_size / batch_size * args.dataloader.balancer_prefetch_factor)
            dataloader_kwargs["batch_size"] = None
            dataloader_kwargs["drop_last"] = None
            dataset = TokenBalancedBatcher(
                dataset,
                batch_size=batch_size,
                buffer_size=args.dataloader.balancer_buffer_size,
            )

        dataloaders[split] = DataLoader(dataset, **dataloader_kwargs)

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


class TokenBalancedBatcher(torch.utils.data.IterableDataset):
    def __init__(self, source_dataset, batch_size=16, buffer_size=2048):
        assert buffer_size % batch_size == 0, "Buffer size must be an integer multiple of batch_size."
        self.source_dataset = source_dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    @property
    def start(self):
        return self.source_dataset.start

    @property
    def end(self):
        return self.source_dataset.end

    @start.setter
    def start(self, value):
        self.source_dataset.start = value

    @end.setter
    def end(self, value):
        self.source_dataset.end = value

    def __iter__(self):
        buffer = []

        for sample in self.source_dataset:
            length = sample["decoder_attention_mask"].sum()
            buffer.append((length, sample))

            if len(buffer) == self.buffer_size:
                yield from self._emit_batches(buffer)
                buffer = []

        if buffer:
            yield from self._emit_batches(buffer)

    def _emit_batches(self, buffer):
        import heapq

        batch_size = self.batch_size
        num_batches = len(buffer) // batch_size
        usable = num_batches * batch_size

        buffer = sorted(buffer[:usable], key=lambda x: x[0], reverse=True)

        batches = [[] for _ in range(num_batches)]
        totals = [0 for _ in range(num_batches)]

        heap = [(0, i) for i in range(num_batches)]
        heapq.heapify(heap)

        for length, sample in buffer:
            total, batch_idx = heapq.heappop(heap)

            batches[batch_idx].append(sample)
            totals[batch_idx] += length

            if len(batches[batch_idx]) < batch_size:
                heapq.heappush(heap, (totals[batch_idx], batch_idx))

        for batch in batches:
            if len(batch) == batch_size:
                yield batch
