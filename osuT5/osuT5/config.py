from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING

from .event import ContextType


# Default config here based on V28

@dataclass
class SpectrogramConfig:
    implementation: str = "nnAudio"  # Spectrogram implementation (nnAudio/torchaudio)
    log_scale: bool = False
    sample_rate: int = 16000
    hop_length: int = 128
    n_fft: int = 1024
    n_mels: int = 388
    f_min: int = 0
    f_max: int = 8000
    pad_mode: str = "constant"


@dataclass
class ModelConfig:
    name: str = "openai/whisper-base"  # Model name
    config_base: str = ""  # Model base for config lookup
    input_features: bool = True
    project_encoder_input: bool = True
    embed_decoder_input: bool = True
    manual_norm_weights: bool = False
    do_style_embed: bool = False
    do_difficulty_embed: bool = False
    do_mapper_embed: bool = False
    do_song_position_embed: bool = False
    cond_dim: int = 128
    cond_size: int = 0
    rope_type: str = "dynamic"  # RoPE type (dynamic/static)
    rope_encoder_scaling_factor: float = 1.0
    rope_decoder_scaling_factor: float = 1.0
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    overwrite: dict = field(default_factory=lambda: {})  # Overwrite model config
    add_config: dict = field(default_factory=lambda: {})  # Add to model config


@dataclass
class DataConfig:
    dataset_type: str = "mmrs"   # Dataset type (ors/mmrs)
    train_dataset_path: str = "/workspace/datasets/MMRS39389"  # Training dataset directory
    train_dataset_start: int = 0  # Training dataset start index
    train_dataset_end: int = 38689  # Training dataset end index
    test_dataset_path: str = "/workspace/datasets/MMRS39389"  # Testing/validation dataset directory
    test_dataset_start: int = 38689  # Testing/validation dataset start index
    test_dataset_end: int = 39389  # Testing/validation dataset end index
    src_seq_len: int = 1024
    tgt_seq_len: int = 2048
    sample_rate: int = 16000
    hop_length: int = 128
    cycle_length: int = 16
    per_track: bool = True  # Loads all beatmaps in a track sequentially which optimizes audio data loading
    only_last_beatmap: bool = False  # Only use the last beatmap in the mapset
    center_pad_decoder: bool = False  # Center pad decoder input
    num_classes: int = 152680
    num_diff_classes: int = 24  # Number of difficulty classes
    max_diff: int = 12  # Maximum difficulty of difficulty classes
    num_cs_classes: int = 21  # Number of circle size classes
    class_dropout_prob: float = 0.2
    diff_dropout_prob: float = 0.2
    mapper_dropout_prob: float = 0.2
    cs_dropout_prob: float = 0.2
    year_dropout_prob: float = 0.2
    hold_note_ratio_dropout_prob: float = 0.2
    scroll_speed_ratio_dropout_prob: float = 0.2
    descriptor_dropout_prob: float = 0.2
    # All Special Prefix Tokens
    add_out_context_types: bool = True  # Add tokens indicating types of the out context
    add_gamemode_token: bool = True
    add_style_token: bool = False
    add_diff_token: bool = True
    add_mapper_token: bool = True
    add_year_token: bool = True
    add_hitsounded_token: bool = True  # Add token for whether the map has hitsounds
    add_song_length_token: bool = True  # Add token for the length of the song
    add_global_sv_token: bool = True  # Add token for the global slider velocity in std and ctb
    add_cs_token: bool = True
    add_keycount_token: bool = True  # Add token for the number of keys in mania
    add_hold_note_ratio_token: bool = True  # Add token for the ratio of hold notes in mania
    add_scroll_speed_ratio_token: bool = True  # Add token for the scroll speed ratio in mania
    add_descriptors: bool = True  # Add beatmap descriptor tokens
    add_sv_special_token: bool = True  # Add token for last SV value
    add_kiai_special_token: bool = True  # Add token for last kiai state
    add_song_position_token: bool = True  # Add token for the position of the song in the mapset
    # ---
    add_empty_sequences: bool = True
    add_empty_sequences_at_step: int = -1
    add_pre_tokens: bool = False
    add_pre_tokens_at_step: int = -1
    max_pre_token_len: int = -1
    timing_random_offset: int = 2
    add_gd_context: bool = False  # Prefix the decoder with tokens of another beatmap in the mapset
    min_difficulty: float = 0  # Minimum difficulty to consider including in the dataset
    sample_weights_path: str = ''  # Path to sample weights
    rhythm_weight: float = 3.0  # Weight of rhythm tokens in the loss calculation
    lookback: float = 0  # Fraction of audio sequence to fill with tokens from previous inference window
    lookahead: float = 0  # Fraction of audio sequence to skip at the end of the audio window
    context_types: list[dict[str, list[ContextType]]] = field(default_factory=lambda: [
        {"in": [ContextType.NONE], "out": [ContextType.TIMING, ContextType.KIAI, ContextType.MAP, ContextType.SV]},
        {"in": [ContextType.NO_HS], "out": [ContextType.TIMING, ContextType.KIAI, ContextType.MAP, ContextType.SV]},
        {"in": [ContextType.GD], "out": [ContextType.TIMING, ContextType.KIAI, ContextType.MAP, ContextType.SV]}
    ])  # List of context types to include in the dataset
    context_weights: list[float] = field(default_factory=lambda: [4, 1, 1])  # List of weights for each context type. Determines how often each context type is sampled
    descriptors_path: str = ''  # Path to file with all beatmap descriptors
    mappers_path: str = ''  # Path to file with all beatmap mappers
    add_timing: bool = False  # Add beatmap timing to map context
    add_snapping: bool = True  # Model hit object snapping
    add_timing_points: bool = True  # Model beatmap timing with timing points
    add_hitsounds: bool = True  # Model beatmap hitsounds
    add_distances: bool = True  # Model hit object distances
    add_positions: bool = True  # Model hit object coordinates
    position_precision: int = 32  # Precision of hit object coordinates
    position_split_axes: bool = False  # Split hit object X and Y coordinates into separate tokens
    position_range: list[int] = field(default_factory=lambda: [-256, 768, -256, 640])  # Range of hit object coordinates
    dt_augment_prob: float = 0.5  # Probability of augmenting the dataset with DT
    dt_augment_range: list[float] = field(default_factory=lambda: [1.25, 1.5])  # Range of DT augmentation
    types_first: bool = True  # Put the type token at the start of the group before the timeshift token
    add_kiai: bool = True  # Add kiai times to map context
    gamemodes: list[int] = field(default_factory=lambda: [0, 1, 2, 3])  # List of gamemodes to include in the dataset
    mania_bpm_normalized_scroll_speed: bool = True  # Normalize mania scroll speed by BPM
    add_sv: bool = True  # Model slider velocity in std and ctb
    add_mania_sv: bool = False  # Add mania scroll velocity in map context
    min_year: Optional[int] = None  # Minimum year of the beatmap to include in the dataset


@dataclass
class DataloaderConfig:
    num_workers: int = 8


@dataclass
class OptimizerConfig:  # Optimizer settings
    name: str = "adamwscale"  # Optimizer
    base_lr: float = 1e-2
    batch_size: int = 128  # Batch size per GPU
    total_steps: int = 65536
    warmup_steps: int = 10000
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_acc: int = 8
    final_cosine: float = 1e-5


@dataclass
class EvalConfig:
    every_steps: int = 1000
    steps: int = 500


@dataclass
class CheckpointConfig:
    every_steps: int = 5000


@dataclass
class LoggingConfig:
    log_with: str = 'wandb'     # Logging service (wandb/tensorboard)
    every_steps: int = 10
    grad_l2: bool = True
    weights_l2: bool = True
    mode: str = 'online'


@dataclass
class ProfileConfig:
    do_profile: bool = False
    early_stop: bool = False
    wait: int = 8
    warmup: int = 8
    active: int = 8
    repeat: int = 1


@dataclass
class TrainConfig:
    compile: bool = True
    device: str = "gpu"
    precision: str = "bf16"
    seed: int = 42
    flash_attention: bool = False
    checkpoint_path: str = ""
    pretrained_path: str = ""
    pretrained_t5_compat: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    hydra: Any = MISSING
    mode: str = "train"


OmegaConf.register_new_resolver("context_type", lambda x: ContextType(x.lower()))
cs = ConfigStore.instance()
cs.store(group="osut5", name="base_train", node=TrainConfig)

