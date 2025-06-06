from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, WhisperForConditionalGeneration, WhisperConfig, \
    PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from .configuration_mapperatorinator import MapperatorinatorConfig
from .custom_transformers import NWhisperConfig, RoPEWhisperConfig, NWhisperForConditionalGeneration, \
    RoPEWhisperForConditionalGeneration
from .spectrogram import MelSpectrogram

LABEL_IGNORE_ID = -100


def get_backbone_model(name, config):
    if name.startswith("google/t5"):
        if isinstance(config, dict):
            config = T5Config(**config)
        model = T5ForConditionalGeneration(config)
    elif name.startswith("OliBomby/nwhisper"):
        if isinstance(config, dict):
            config = NWhisperConfig(**config)
        model = NWhisperForConditionalGeneration(config)
    elif name.startswith("Tiger14n/ropewhisper"):
        if isinstance(config, dict):
            config = RoPEWhisperConfig(**config)
        model = RoPEWhisperForConditionalGeneration(config)
    elif name.startswith("openai/whisper"):
        if isinstance(config, dict):
            config = WhisperConfig(**config)
        model = WhisperForConditionalGeneration(config)
    else:
        raise NotImplementedError

    return model


class Mapperatorinator(PreTrainedModel, GenerationMixin):
    __slots__ = ["spectrogram", "decoder_embedder", "encoder_embedder", "transformer", "style_embedder", "num_classes"]
    config_class = MapperatorinatorConfig
    base_model_prefix = "model"
    main_input_name = "frames"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: MapperatorinatorConfig):
        super().__init__(config)

        self.spectrogram = MelSpectrogram(
            config.spectrogram_implementation,
            config.spectrogram_log_scale,
            config.sample_rate,
            config.n_fft,
            config.n_mels,
            config.hop_length,
            config.f_min,
            config.f_max,
            config.pad_mode,
        )

        self.transformer: WhisperForConditionalGeneration = get_backbone_model(config.backbone_model_name, config.backbone_config)

        self.num_classes = config.num_classes
        self.input_features = config.input_features
        self.project_encoder_input = config.project_encoder_input
        self.embed_decoder_input = config.embed_decoder_input

        self.do_style_embed = config.do_style_embed
        self.do_difficulty_embed = config.do_difficulty_embed
        self.do_mapper_embed = config.do_mapper_embed
        self.do_song_position_embed = config.do_song_position_embed
        d_model = config.hidden_size

        if self.do_style_embed:
            self.style_embedder = LabelEmbedder(self.num_classes, d_model)
            nn.init.normal_(self.style_embedder.embedding_table.weight, std=config.init_std)

        if self.do_difficulty_embed:
            self.difficulty_embedder = DifficultyEmbedder(
                hidden_size=config.cond_dim,
                max_difficulty=10,
            )

        if self.do_mapper_embed:
            self.mapper_embedder = MapperStyleEmbedder(
                embedding_dim=config.cond_dim,
                num_mappers=config.num_mappers,
            )

        if self.do_song_position_embed:
            self.song_pos_embedder = SongPositionEmbedder(
                hidden_size=config.cond_dim,
                num_basis=10,
            )

        if self.project_encoder_input:
            self.encoder_embedder = nn.Linear(config.n_mels + config.cond_size, d_model)

        if self.embed_decoder_input:
            self.decoder_embedder = nn.Embedding(config.vocab_size_in, d_model)
            self.decoder_embedder.weight.data.normal_(mean=0.0, std=1.0)

        class_weights = torch.ones(config.vocab_size)
        class_weights[config.rhythm_token_start:config.rhythm_token_end] = config.rhythm_weight
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=LABEL_IGNORE_ID)

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            beatmap_idx: Optional[torch.Tensor] = None,
            difficulty: Optional[torch.Tensor] = None,
            mapper_idx: Optional[torch.Tensor] = None,
            song_position: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            sample_weights: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> Seq2SeqLMOutput:
        """
        frames: B x L_encoder x mel_bins, float32
        decoder_input_ids: B x L_decoder, int64
        beatmap_idx: B, int64
        beatmap_id: B, int64
        encoder_outputs: B x L_encoder x D, float32
        """
        if beatmap_idx is None and self.do_style_embed:
            batch_size = frames.shape[0] if frames is not None else decoder_input_ids.shape[0]
            device = frames.device if frames is not None else decoder_input_ids.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        inputs_embeds = None
        if encoder_outputs is None and frames is not None:
            frames = self.spectrogram(frames)  # (N, L, M)
            conds = []

            if self.do_style_embed:
                style_embedding = self.style_embedder(beatmap_idx)  # (N, D)
                style_embedding = style_embedding.unsqueeze(1).repeat((1, frames.shape[1], 1))
                conds.append(style_embedding)
            if self.do_difficulty_embed:
                difficulty_embedding = self.difficulty_embedder(difficulty)
                conds.append(difficulty_embedding)
            if self.do_mapper_embed:
                mapper_embedding = self.mapper_embedder(mapper_idx)
                conds.append(mapper_embedding)
            if self.do_song_position_embed:
                song_position_embedding = self.song_pos_embedder(song_position)
                conds.append(song_position_embedding)

            conds_expanded = [c.unsqueeze(1).expand((-1, frames.shape[1], -1)) for c in conds]
            inputs_embeds = torch.concatenate([frames] + conds_expanded, dim=-1)

        inputs = dict(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, labels=labels, **kwargs
        )

        if self.project_encoder_input:
            inputs_embeds = self.encoder_embedder(inputs_embeds) if inputs_embeds is not None else None

        if self.input_features:
            inputs["input_features"] = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None
            del inputs["inputs_embeds"]

        if self.embed_decoder_input:
            inputs["decoder_inputs_embeds"] = self.decoder_embedder(decoder_input_ids)
            del inputs["decoder_input_ids"]

        output = self.transformer.forward(**inputs)

        if labels is not None:
            unreduced_loss = self.loss_fn(torch.swapaxes(output.logits, 1, -1), labels)
            if sample_weights is not None:
                unreduced_loss *= sample_weights.unsqueeze(1)
            output.loss = unreduced_loss.sum() / (labels != LABEL_IGNORE_ID).sum()

        return output

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        beatmap_idx=None,
        decoder_attention_mask=None,
        cache_position=None,
        negative_prompt=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        # Add negative prompt to the input for classifier free guidance
        if negative_prompt is not None:
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            decoder_input_ids[:decoder_input_ids.shape[0] // 2, :negative_prompt.shape[1]] = negative_prompt

            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))
                if negative_prompt_attention_mask is not None:
                    decoder_attention_mask[:decoder_attention_mask.shape[0] // 2, :negative_prompt_attention_mask.shape[1]] = negative_prompt_attention_mask

            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state.repeat((2, 1, 1)))

        inputs = self.transformer.prepare_inputs_for_generation(
            input_ids=decoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        inputs["beatmap_idx"] = beatmap_idx
        return inputs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: torch.device = None,
    ):
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        if device is None:
            device = self.device
        if decoder_start_token_id.ndim == 1:
            if decoder_start_token_id.shape[0] != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {decoder_start_token_id.shape[0]}"
                )
            decoder_start_token_id = decoder_start_token_id.view(-1, 1)
        else:
            decoder_start_token_id = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # Mapperatorinator handles the task-specific decoder input externally
        if decoder_input_ids is None:
            decoder_input_ids = decoder_start_token_id

        return decoder_input_ids, model_kwargs

    def can_generate(self) -> bool:
        return True

    def get_encoder(self):
        return OsuTEncoder(
            self.transformer.get_encoder(),
            self.spectrogram,
            self.style_embedder if self.do_style_embed else None,
            self.difficulty_embedder if self.do_difficulty_embed else None,
            self.mapper_embedder if self.do_mapper_embed else None,
            self.song_pos_embedder if self.do_song_position_embed else None,
            self.encoder_embedder if self.project_encoder_input else None,
            self.num_classes,
            self.input_features,
            self.project_encoder_input,
            self.do_style_embed,
            self.do_difficulty_embed,
            self.do_mapper_embed,
            self.do_song_position_embed,
        )

    def get_decoder(self):
        return self.transformer.get_decoder()


class OsuTEncoder(nn.Module):
    def __init__(
            self,
            base_encoder: nn.Module,
            spectrogram: MelSpectrogram,
            style_embedder: LabelEmbedder,
            difficulty_embedder: DifficultyEmbedder,
            mapper_embedder: MapperStyleEmbedder,
            song_pos_embedder: SongPositionEmbedder,
            encoder_embedder: nn.Linear,
            num_classes: int,
            input_features: bool,
            project_encoder_input: bool,
            do_style_embed: bool,
            do_difficulty_embed: bool,
            do_mapper_embed: bool,
            do_song_position_embed: bool,
    ):
        super().__init__()
        self.base = base_encoder
        self.spectrogram = spectrogram
        self.style_embedder = style_embedder
        self.difficulty_embedder = difficulty_embedder
        self.mapper_embedder = mapper_embedder
        self.song_pos_embedder = song_pos_embedder
        self.encoder_embedder = encoder_embedder
        self.num_classes = num_classes
        self.input_features = input_features
        self.project_encoder_input = project_encoder_input
        self.do_style_embed = do_style_embed
        self.do_difficulty_embed = do_difficulty_embed
        self.do_mapper_embed = do_mapper_embed
        self.do_song_position_embed = do_song_position_embed

    def forward(
            self,
            frames: torch.FloatTensor,
            beatmap_idx: Optional[torch.Tensor] = None,
            difficulty: Optional[torch.Tensor] = None,
            mapper_idx: Optional[torch.Tensor] = None,
            song_position: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False
    ):
        if beatmap_idx is None and self.do_style_embed:
            batch_size = frames.shape[0]
            device = frames.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        frames = self.spectrogram(frames)  # (N, L, M)
        conds = []

        if self.do_style_embed:
            style_embedding = self.style_embedder(beatmap_idx)  # (N, D)
            style_embedding = style_embedding.unsqueeze(1).repeat((1, frames.shape[1], 1))
            conds.append(style_embedding)
        if self.do_difficulty_embed:
            difficulty_embedding = self.difficulty_embedder(difficulty)
            conds.append(difficulty_embedding)
        if self.do_mapper_embed:
            mapper_embedding = self.mapper_embedder(mapper_idx)
            conds.append(mapper_embedding)
        if self.do_song_position_embed:
            song_position_embedding = self.song_pos_embedder(song_position)
            conds.append(song_position_embedding)

        conds_expanded = [c.unsqueeze(1).expand((-1, frames.shape[1], -1)) for c in conds]
        inputs_embeds = torch.concatenate([frames] + conds_expanded, dim=-1)

        if self.project_encoder_input:
            inputs_embeds = self.encoder_embedder(inputs_embeds) if inputs_embeds is not None else None

        if self.input_features:
            inputs_embeds = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None

        return self.base.forward(
            inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(
            num_classes + 1,
            hidden_size,
        )

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class DifficultyEmbedder(nn.Module):
    def __init__(self, hidden_size=64, max_difficulty=10.0, num_basis=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_difficulty = max_difficulty
        self.num_basis = num_basis

        # Learnable basis centers
        self.register_parameter(
            'basis_centers',
            nn.Parameter(torch.linspace(0, 1, num_basis))
        )

        # Learnable basis widths
        self.register_parameter(
            'basis_widths',
            nn.Parameter(torch.ones(num_basis) * 0.1)
        )

        self.difficulty_proj = nn.Sequential(
            nn.Linear(num_basis, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Initialize with smaller weights
        for m in self.difficulty_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Reduce gain
                nn.init.zeros_(m.bias)

    def compute_basis_functions(self, diff_normalized):
        # Compute RBF basis functions
        diff_expanded = diff_normalized.unsqueeze(-1)  # [B, 1]
        centers = self.basis_centers.view(1, -1)       # [1, N]
        widths = self.basis_widths.view(1, -1)        # [1, N]

        # Gaussian RBF
        basis = torch.exp(
            -(diff_expanded - centers).pow(2) / (2 * widths.pow(2))
        )
        return basis

    def forward(self, difficulty):
        # Normalize difficulty
        diff_normalized = difficulty / self.max_difficulty

        # Compute basis functions
        basis = self.compute_basis_functions(diff_normalized)

        # Project to embedding space
        return self.difficulty_proj(basis)


class MapperStyleEmbedder(nn.Module):
    """
    Embedding layer for mapper styles
    """
    def __init__(self, num_mappers: int, embedding_dim: int = 64, dropout_prob: float = 0.1):
        """
        Args:
            num_mappers: Total number of unique mappers.
            embedding_dim: Size of the embedding vector.
            dropout_prob: Dropout probability for regularization.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_mappers = num_mappers

        # Embedding table: num_mappers rows for actual mappers + 1 row for default style (-1)
        self.embedding = nn.Embedding(num_embeddings=num_mappers + 1, embedding_dim=embedding_dim)

        # Initialize embeddings with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Dropout for regularization to help small mappers generalize
        self.dropout = nn.Dropout(p=dropout_prob)

        # Layer normalization to stabilize embeddings (especially for mappers with few maps)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, mapper_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            mapper_ids: Tensor of shape [B] with mapper IDs (long/int), where:
                - IDs >= 0 correspond to specific mappers (0 to num_mappers-1).
                - ID = -1 triggers the default style.

        Returns:
            Embedding tensor of shape [B, embedding_dim] if mapper_ids is provided,
        """
        if mapper_ids is None:
            return None  # No conditioning applied

        # Map -1 to the last index (default style) and ensure IDs are valid
        mapper_ids = torch.where(
            mapper_ids == -1,
            torch.tensor(self.num_mappers, device=mapper_ids.device),
            mapper_ids
        )

        # Ensure mapper_ids are within bounds (0 to num_mappers)
        mapper_ids = torch.clamp(mapper_ids, min=0, max=self.num_mappers)

        # Retrieve embeddings: [B] -> [B, embedding_dim]
        embeddings = self.embedding(mapper_ids)

        # Apply dropout and normalization
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)

        return embeddings


class SongPositionEmbedder(nn.Module):
    """
    Generates an embedding vector representing the global position and duration
    context of an audio chunk within a larger song.

    It takes a normalized start and end position of an audio chunk
    (e.g., [0.25, 0.30] meaning the chunk covers 25% to 30% of the total song)

    This allows the model to be aware of:
    - Where the current audio chunk begins within the song.
    - Where the current audio chunk ends within the song.
    - Implicitly, the duration or extent of the chunk relative to the song.
    This information can help the model make decisions appropriate for different
    song sections (e.g., intro, verse, chorus, outro) and varying chunk lengths.
    """
    def __init__(self, hidden_size=64, num_basis=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_basis = num_basis

        # Learnable basis centers from 0 to 1
        self.register_parameter(
            'basis_centers',
            nn.Parameter(torch.linspace(0, 1, num_basis))
        )

        # Learnable basis widths
        self.register_parameter(
            'basis_widths',
            nn.Parameter(torch.ones(num_basis) * 0.1)
        )

        self.position_proj = nn.Sequential(
            nn.Linear(num_basis * 2, hidden_size * 2),  # start and end positions
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),  # Reduce back to original hidden size
            nn.LayerNorm(hidden_size),
        )


        # Initialize weights
        for m in self.position_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def compute_basis_functions(self, position):
        # Compute RBF basis functions
        position_expanded = position.unsqueeze(-1)  # [B, 1]
        centers = self.basis_centers.view(1, -1)    # [1, N]
        widths = self.basis_widths.view(1, -1)      # [1, N]

        # Gaussian RBF
        basis = torch.exp(
            -(position_expanded - centers).pow(2) / (2 * widths.pow(2))
        )
        return basis

    def forward(self, position_range):
        """
        Args:
            position_range: Tensor of shape [B, 2] containing normalized start and end positions
                            position_range[:, 0] is the start position (0 to 1)
                            position_range[:, 1] is the end position (0 to 1)
        """
        # Split start and end positions
        start_pos = position_range[:, 0]
        end_pos = position_range[:, 1]

        # Compute basis functions for both positions
        start_basis = self.compute_basis_functions(start_pos)  # [B, num_basis]
        end_basis = self.compute_basis_functions(end_pos)     # [B, num_basis]

        # Concatenate bases
        combined_basis = torch.cat([start_basis, end_basis], dim=1)  # [B, num_basis*2]

        # Project to embedding space
        return self.position_proj(combined_basis)
