from transformers import PretrainedConfig, T5Config, WhisperConfig

from .custom_transformers import NWhisperConfig, RoPEWhisperConfig


class MapperatorinatorConfig(PretrainedConfig):
    model_type = "mapperatorinator"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        backbone_model_name: str = "openai/whisper-base",
        backbone_overwrite: dict = None,
        backbone_add_config: dict = None,
        flash_attention: bool = False,
        vocab_size_in=9920,
        vocab_size_out=3988,
        num_classes: int = 0,
        num_mappers: int = 3731,
        input_features: bool = True,
        project_encoder_input: bool = True,
        embed_decoder_input: bool = True,
        do_style_embed: bool = False,
        do_difficulty_embed: bool = False,
        do_mapper_embed: bool = False,
        do_song_position_embed: bool = False,
        cond_dim=128,
        cond_size=0,
        spectrogram_implementation: str = "nnAudio",
        spectrogram_log_scale: bool = False,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        n_mels: int = 388,
        hop_length: int = 128,
        f_min: int = 0,
        f_max: int = 8000,
        pad_mode: str = "constant",
        rhythm_weight: float = 3.0,
        rhythm_token_start: int = 17,
        rhythm_token_end: int = 836,
        label_smoothing: float = 0.0,
        init_std=0.02,
        src_seq_len=1024,
        tgt_seq_len=2048,
        rope_type="dynamic",
        rope_encoder_scaling_factor=1.0,
        rope_decoder_scaling_factor=1.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=1,
        do_sample=True,
        top_k=0,
        max_length=2048,
        **kwargs,
    ):
        # Get backbone model config
        if backbone_model_name.startswith("google/t5"):
            config = T5Config.from_pretrained(backbone_model_name)
        elif backbone_model_name.startswith("openai/whisper"):
            config = WhisperConfig.from_pretrained(backbone_model_name)
        elif backbone_model_name.startswith("OliBomby/nwhisper"):
            config = NWhisperConfig.from_pretrained("openai/whisper" + backbone_model_name[17:])
        elif backbone_model_name.startswith("Tiger14n/ropewhisper"):
            config = RoPEWhisperConfig.from_pretrained("openai/whisper" + backbone_model_name[20:])
        else:
            raise NotImplementedError

        config.vocab_size = vocab_size_out
        config.use_cache = False

        if backbone_overwrite is not None:
            for k, v in backbone_overwrite.items():
                assert hasattr(config, k), f"config does not have attribute {k}"
                setattr(config, k, v)

        if backbone_add_config is not None:
            for k, v in backbone_add_config.items():
                assert not hasattr(config, k), f"config already has attribute {k}"
                setattr(config, k, v)

        if isinstance(config, WhisperConfig):
            # Includes NWhisperConfig and RoPEWhisperConfig
            config.num_mel_bins = config.d_model if project_encoder_input else n_mels + cond_size
            config.pad_token_id = pad_token_id
            config.bos_token_id = bos_token_id
            config.eos_token_id = eos_token_id
            config.max_source_positions = src_seq_len // 2
            config.max_target_positions = tgt_seq_len
            config.begin_suppress_tokens = None
            config.decoder_start_token_id = bos_token_id
            config.forced_decoder_ids = None
            config.suppress_tokens = None
            if flash_attention:
                config._attn_implementation = "flash_attention_2"
        if isinstance(config, NWhisperConfig):
            config.input_vocab_size = vocab_size_in
        if isinstance(config, RoPEWhisperConfig):
            config.rope_type = rope_type
            config.rope_encoder_scaling_factor = rope_encoder_scaling_factor
            config.rope_decoder_scaling_factor = rope_decoder_scaling_factor

        self.backbone_model_name = backbone_model_name
        self.backbone_config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.max_source_positions = config.max_source_positions
        self.max_target_positions = config.max_target_positions
        self.vocab_size_in = vocab_size_in
        self.vocab_size = vocab_size_out
        self.num_classes = num_classes
        self.num_mappers = num_mappers
        self.input_features = input_features
        self.project_encoder_input = project_encoder_input
        self.embed_decoder_input = embed_decoder_input
        self.do_style_embed = do_style_embed
        self.do_difficulty_embed = do_difficulty_embed
        self.do_mapper_embed = do_mapper_embed
        self.do_song_position_embed = do_song_position_embed
        self.cond_dim = cond_dim
        self.cond_size = cond_size
        self.spectrogram_implementation = spectrogram_implementation
        self.spectrogram_log_scale = spectrogram_log_scale
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.pad_mode = pad_mode
        self.rhythm_weight = rhythm_weight
        self.rhythm_token_start = rhythm_token_start
        self.rhythm_token_end = rhythm_token_end
        self.label_smoothing = 0.0
        self.init_std = init_std
        self.disable_compile = True

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            do_sample=do_sample,
            top_k=top_k,
            max_length=max_length,
            **kwargs,
        )
