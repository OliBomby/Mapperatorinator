from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# Default config here based on V28


@dataclass
class ModelConfig:
    model: str = "DiT-B"  # Model name
    noise_schedule: str = 'squaredcos_cap_v2'  # Noise schedule
    l1_loss: bool = False  # L1 loss
    diffusion_steps: int = 1000  # Number of diffusion steps
    max_diffusion_step: int = 100  # Maximum diffusion step used in training. Reduce to make a specialized refinement model
    context_size: int = 272  # Size of the context vector fed to the model


@dataclass
class DataConfig:
    train_dataset_path: str = '/workspace/datasets/ORS16291/'  # Path to the data
    start: int = 0
    end: int = 16291
    shuffle: bool = True
    seq_len: int = 128  # Sequence length
    stride: int = 16  # Stride
    cycle_length: int = 64  # Cycle length
    beatmap_class: bool = False  # Include beatmap classes
    difficulty_class: bool = True  # Include difficulty classes
    mapper_class: bool = True  # Include mapper classes
    descriptor_class: bool = True  # Include descriptor classes
    circle_size_class: bool = True  # Include circle size classes
    class_dropout_prob: float = 0.2
    diff_dropout_prob: float = 0.2
    mapper_dropout_prob: float = 0.2
    descriptor_dropout_prob: float = 0.2
    cs_dropout_prob: float = 0.2
    descriptors_path: str = "../../../datasets/beatmap_descriptors.csv"  # Path to file with all beatmap descriptors
    mappers_path: str = "../../../datasets/beatmap_users.json"  # Path to file with all beatmap mappers
    num_diff_classes: int = 26  # Number of difficulty classes
    max_diff: int = 12  # Maximum difficulty of difficulty classes
    num_cs_classes: int = 22  # Number of circle size classes
    double_time_prob: float = 0.5
    distance_std: float = 0.1  # Standard deviation of the distance noise


@dataclass
class DataloaderConfig:
    num_workers: int = 4


@dataclass
class OptimizerConfig:  # Optimizer settings
    name: str = "adamw"  # Optimizer
    base_lr: float = 2e-4  # Base learning rate
    batch_size: int = 256  # Global batch size
    total_steps: int = 400000
    warmup_steps: int = 10000
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    grad_acc: int = 2
    grad_clip: float = 1.0
    final_cosine: float = 1e-5


@dataclass
class CheckpointConfig:
    every_steps: int = 5000


@dataclass
class LoggingConfig:
    log_with: str = 'wandb'     # Logging service (wandb/tensorboard)
    every_steps: int = 10
    mode: str = 'online'


@dataclass
class DiffusionTrainConfig:
    compile: bool = True
    device: str = "gpu"
    precision: str = "bf16"
    seed: int = 0
    checkpoint_path: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hydra: Any = MISSING


cs = ConfigStore.instance()
cs.store(group="diffusion", name="base", node=DiffusionTrainConfig)
