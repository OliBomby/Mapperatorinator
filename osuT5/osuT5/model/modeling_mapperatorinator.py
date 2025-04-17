from __future__ import annotations

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, WhisperForConditionalGeneration, WhisperConfig, \
    PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .configuration_mapperatorinator import MapperatorinatorConfig
from .configuration_nwhisper import NWhisperConfig
from .modeling_nwhisper import NWhisperForConditionalGeneration
from ..model.spectrogram import MelSpectrogram

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
    elif name.startswith("openai/whisper"):
        if isinstance(config, dict):
            config = WhisperConfig(**config)
        config._attn_implementation = "sdpa"
        model = WhisperForConditionalGeneration(config)
    else:
        raise NotImplementedError

    return model


class Mapperatorinator(PreTrainedModel):
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

        self.spectrogram = MelSpectrogram(config.sample_rate, config.n_fft, config.n_mels, config.hop_length)
        self.transformer: WhisperForConditionalGeneration = get_backbone_model(config.backbone_model_name, config.backbone_config)

        self.num_classes = config.num_classes
        self.input_features = config.input_features
        self.embed_decoder_input = config.embed_decoder_input
        self.do_style_embed = config.do_style_embed
        d_model = config.hidden_size

        if self.embed_decoder_input:
            self.decoder_embedder = nn.Embedding(config.vocab_size_in, d_model)
            self.decoder_embedder.weight.data.normal_(mean=0.0, std=1.0)

        if self.do_style_embed:
            self.style_embedder = LabelEmbedder(self.num_classes, d_model)
            self.encoder_embedder = nn.Linear(config.n_mels + d_model, d_model)
            nn.init.normal_(self.style_embedder.embedding_table.weight, std=config.init_std)
        else:
            self.encoder_embedder = nn.Linear(config.n_mels, d_model)

        class_weights = torch.ones(config.vocab_size)
        class_weights[config.rhythm_token_start:config.rhythm_token_end] = config.rhythm_weight
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=LABEL_IGNORE_ID)

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            beatmap_idx: Optional[torch.Tensor] = None,
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
            if self.do_style_embed:
                style_embeds = self.style_embedder(beatmap_idx)  # (N, D)
                frames_concat = torch.concatenate((frames, style_embeds.unsqueeze(1).expand((-1, frames.shape[1], -1))), -1)
                inputs_embeds = self.encoder_embedder(frames_concat)
            else:
                inputs_embeds = self.encoder_embedder(frames)

        inputs = dict(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs, labels=labels, **kwargs
        )

        if self.embed_decoder_input:
            inputs["decoder_inputs_embeds"] = self.decoder_embedder(decoder_input_ids)
            del inputs["decoder_input_ids"]

        if self.input_features:
            inputs["input_features"] = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None
            del inputs["inputs_embeds"]

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
            self.encoder_embedder,
            self.num_classes,
            self.input_features,
            self.do_style_embed
        )

    def get_decoder(self):
        return self.transformer.get_decoder()


class OsuTEncoder(nn.Module):
    def __init__(
            self,
            base_encoder: WhisperEncoder,
            spectrogram: MelSpectrogram,
            style_embedder: LabelEmbedder,
            encoder_embedder: nn.Linear,
            num_classes: int,
            input_features: bool,
            do_style_embed: bool
    ):
        super().__init__()
        self.base = base_encoder
        self.spectrogram = spectrogram
        self.style_embedder = style_embedder
        self.encoder_embedder = encoder_embedder
        self.num_classes = num_classes
        self.input_features = input_features
        self.do_style_embed = do_style_embed

    def forward(
            self,
            frames: torch.FloatTensor,
            beatmap_idx: torch.Tensor,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = False
    ):
        if beatmap_idx is None and self.do_style_embed:
            batch_size = frames.shape[0]
            device = frames.device
            beatmap_idx = torch.full([batch_size], self.num_classes, dtype=torch.long, device=device)

        frames = self.spectrogram(frames)  # (N, L, M)
        if self.do_style_embed:
            style_embeds = self.style_embedder(beatmap_idx)  # (N, D)
            frames_concat = torch.concatenate((frames, style_embeds.unsqueeze(1).expand((-1, frames.shape[1], -1))),
                                              -1)
            inputs_embeds = self.encoder_embedder(frames_concat)
        else:
            inputs_embeds = self.encoder_embedder(frames)

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
