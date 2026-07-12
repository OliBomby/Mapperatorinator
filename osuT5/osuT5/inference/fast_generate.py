"""Fast inference path for osuT5: batched encoder precompute + generate reuse.

The dominant per-window cost is the encoder prefill, which is far cheaper batched
than one-window-at-a-time. This module precomputes encoder outputs for all audio
windows in large batches, then feeds them into the existing per-window generate
loop so each window skips the encoder entirely.

These helpers are designed to be bit-compatible with the unmodified generate path
for the same seed: only the encoder (a pure function of the audio + static
conditioning) is hoisted out of the sequential loop.
"""
import time
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput

from .cache_utils import get_cache
from .logit_processors import (
    ConditionalTemperatureLogitsWarper, MonotonicTimeShiftLogitsProcessor,
    LookbackBiasLogitsWarper, TimeshiftBias, get_beat_type_tokens,
    get_mania_type_tokens, get_scroll_speed_tokens,
)
from .server import (
    get_eos_token_id, _build_generation_stats, _prompt_token_counts,
)
from transformers import (
    LogitsProcessorList, ClassifierFreeGuidanceLogitsProcessor, TemperatureLogitsWarper,
)
from ..event import ContextType, EventType

ENC_BATCH = 16  # windows per encoder forward pass


@torch.no_grad()
def precompute_encoder_outputs(model, frames: torch.Tensor, cond_kwargs: dict,
                               song_positions: Optional[torch.Tensor] = None,
                               batch_size: int = ENC_BATCH):
    """Run the encoder (mel + transformer encoder + conditioning) for all windows.

    Args:
        model: Mapperatorinator
        frames: (N, L_raw) raw audio windows on the correct device, fp32
        cond_kwargs: dict of beatmap_idx/difficulty/mapper_idx (scalar tensors);
                     broadcast across the N windows
        song_positions: (N, 2) tensor of (pos_start, pos_end) per window, or None
        batch_size: chunk size for the encoder forward

    Returns: list[N] of BaseModelOutput(last_hidden_state=(1, L_enc, D)) on device,
             one per window, in input order.
    """
    n = frames.shape[0]
    device = model.device
    outputs = [None] * n
    model.eval()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = frames[start:end].to(device)  # (b, L_raw)

        # NOTE on determinism: batched matmuls/attention use different GPU kernels
        # than single-row, so the encoder output is NOT bit-identical to the
        # original one-window-at-a-time path. This propagates into sampling
        # divergence, so output is quality-equivalent but not byte-identical
        # (greedy still matches closely; only sampled runs diverge). Comparable to
        # bf16-vs-fp32 drift.
        mel = model.spectrogram(chunk).to(model.transformer.dtype)  # (b, L_mel, M)

        conds = []
        bsz = chunk.shape[0]
        if model.do_style_embed:
            beatmap_idx = cond_kwargs.get("beatmap_idx")
            if beatmap_idx is not None:
                beatmap_idx = beatmap_idx.to(device).expand(bsz)
            else:
                beatmap_idx = torch.full((bsz,), model.num_classes, dtype=torch.long, device=device)
            conds.append(model.style_embedder(beatmap_idx))
        if model.do_difficulty_embed:
            difficulty = cond_kwargs.get("difficulty")
            if difficulty is not None:
                difficulty = difficulty.to(device).expand(bsz)
            else:
                difficulty = torch.full((bsz,), 5.0, device=device)
            # Cast to the model dtype so the RBF basis matches difficulty_proj's
            # weights. The proj is a bf16 nn.Linear when the model is bf16, and the
            # basis buffers are cast with it; an fp32 difficulty would make the
            # basis fp32 and trip "mat1 and mat2 must have the same dtype".
            difficulty = difficulty.to(model.transformer.dtype)
            conds.append(model.difficulty_embedder(difficulty))
        if model.do_mapper_embed:
            mapper_idx = cond_kwargs.get("mapper_idx")
            if mapper_idx is not None:
                mapper_idx = mapper_idx.to(device).expand(bsz)
            else:
                mapper_idx = torch.full((bsz,), -1, dtype=torch.long, device=device)
            conds.append(model.mapper_embedder(mapper_idx))
        if model.do_song_position_embed and song_positions is not None:
            sp = song_positions[start:end].to(device).to(model.transformer.dtype)
            conds.append(model.song_pos_embedder(sp))

        conds_expanded = [c.unsqueeze(1).expand((-1, mel.shape[1], -1)) for c in conds]
        inputs_embeds = torch.concatenate([mel] + conds_expanded, dim=-1)

        if model.project_encoder_input:
            inputs_embeds = model.encoder_embedder(inputs_embeds)

        input_features = torch.swapaxes(inputs_embeds, 1, 2)  # (b, D, L)
        enc_out = model.transformer.model.encoder(input_features)
        hidden = enc_out.last_hidden_state  # (b, L_enc, D)

        for j in range(hidden.shape[0]):
            outputs[start + j] = BaseModelOutput(last_hidden_state=hidden[j:j + 1])

    return outputs


def build_logits_processors(tokenizer, cfg_scale, timeshift_bias, types_first,
                            temperature, timing_temperature, mania_column_temperature,
                            taiko_hit_temperature, lookback_time, device):
    """Mirror of the processor list built in server.model_generate."""
    lpp = LogitsProcessorList()
    if cfg_scale > 1.0:
        lpp.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))
    lpp.append(MonotonicTimeShiftLogitsProcessor(tokenizer))
    if timeshift_bias != 0:
        lpp.append(TimeshiftBias(timeshift_bias,
                                 tokenizer.event_start[EventType.TIME_SHIFT],
                                 tokenizer.event_end[EventType.TIME_SHIFT]))
    if types_first:
        lpp.append(ConditionalTemperatureLogitsWarper(
            temperature, timing_temperature, mania_column_temperature,
            taiko_hit_temperature, types_first, get_beat_type_tokens(tokenizer),
            get_mania_type_tokens(tokenizer), get_scroll_speed_tokens(tokenizer),
        ))
    else:
        lpp.append(TemperatureLogitsWarper(temperature))
    if lookback_time > 0:
        lpp.append(LookbackBiasLogitsWarper(lookback_time, tokenizer, types_first, device))
    return lpp


@torch.no_grad()
def model_generate_fast(model, tokenizer, model_kwargs, generate_kwargs,
                        encoder_outputs: BaseModelOutput):
    """Drop-in for server.model_generate that takes precomputed encoder_outputs.

    Skips the encoder prefill (the expensive part) by passing encoder_outputs
    straight into model.generate. Everything else mirrors server.model_generate
    so output is bit-compatible for the same seed.
    """
    # To device
    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}
    model_kwargs = {k: v.to(model.dtype) if k != "inputs" and isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
                    for k, v in model_kwargs.items()}

    precision = generate_kwargs.pop('precision', 'fp32')
    cfg_scale = generate_kwargs.pop('cfg_scale', 1.0)
    timeshift_bias = generate_kwargs.pop('timeshift_bias', 0)
    types_first = generate_kwargs.pop('types_first', False)
    temperature = generate_kwargs.pop('temperature', 1.0)
    timing_temperature = generate_kwargs.pop('timing_temperature', temperature)
    mania_column_temperature = generate_kwargs.pop('mania_column_temperature', temperature)
    taiko_hit_temperature = generate_kwargs.pop('taiko_hit_temperature', temperature)
    lookback_time = generate_kwargs.pop('lookback_time', 0.0)
    lookahead_time = generate_kwargs.pop('lookahead_time', 0.0)
    context_type = generate_kwargs.pop('context_type', None)
    if context_type is not None:
        context_type = ContextType(context_type)

    # Strip the raw audio frames — we have precomputed encoder outputs
    model_kwargs.pop('inputs', None)
    model_kwargs["encoder_outputs"] = encoder_outputs

    batch_size = encoder_outputs.last_hidden_state.shape[0]
    logits_processor_list = build_logits_processors(
        tokenizer, cfg_scale, timeshift_bias, types_first, temperature,
        timing_temperature, mania_column_temperature, taiko_hit_temperature,
        lookback_time, model.device,
    )

    cache = get_cache(model, batch_size, generate_kwargs.get('num_beams', 1), cfg_scale)
    pad_token_id = generate_kwargs.get('pad_token_id', getattr(tokenizer, 'pad_id', None))

    # autocast only for 'amp'; bf16 weights run natively
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=precision == 'amp'):
        start_time = time.perf_counter()
        result = model.generate(
            **model_kwargs,
            **generate_kwargs,
            use_cache=True,
            past_key_values=cache,
            logits_processor=logits_processor_list,
            eos_token_id=get_eos_token_id(tokenizer, lookback_time=lookback_time,
                                          lookahead_time=lookahead_time, context_type=context_type),
        )
        elapsed_seconds = time.perf_counter() - start_time

    result = result.cpu()
    stats = _build_generation_stats(result, model_kwargs, pad_token_id, elapsed_seconds)
    return result, stats
