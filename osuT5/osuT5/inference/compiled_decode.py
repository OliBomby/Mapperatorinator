"""Custom decode loop for osuT5 using direct CUDA graph capture.

model.generate() rebuilds tensors and runs host-syncing logits processors every
step, which leaves this small decoder GPU-underutilised. This module replaces it
with a tight loop:
  1. prefill  — eager, processes the prompt, fills the StaticCache
  2. decode   — the single-token forward is captured into a CUDA graph and
                replayed per token. prepare_inputs runs on CPU between replays
                (cheap), logits processors + top-p sampling run in Python.

The decode graph is shape-static (batch, 1 token) and is captured ONCE then
reused across all windows. The encoder hidden state is a static buffer rewritten
per window; the StaticCache is reset (not reallocated) per window.

Output is quality-equivalent (not bit-identical) to HF generate: the sampling RNG
draw order differs, and (when enabled) the batched encoder precompute perturbs the
encoder hidden states at the last bit, so sampled tokens can diverge. Greedy
decoding matches HF token-for-token.
"""
import time
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers import LogitsProcessorList

from .cache_utils import get_cache
from .server import get_eos_token_id, _build_generation_stats
from ..event import ContextType, EventType
from .fast_generate import build_logits_processors
from .logit_processors import MonotonicTimeShiftLogitsProcessor


def neutralize_dynamic_rope(model) -> int:
    """Make dynamic-RoPE decoders (e.g. Mapperatorinator-v30) CUDA-graph capturable.

    Dynamic RoPE runs `_dynamic_frequency_update` inside the forward, which does a
    data-dependent branch (`if seq_len > max_seq_len_cached`) and a buffer mutation.
    Both a device->host sync and in-capture state mutation are illegal during graph
    capture. But the branch only fires when the sequence grows past the rotary
    embedding's cached length, and that cache is sized to `max_target_positions`
    (see RoPEWhisperDecoder), which also caps generation length. So during decode
    the branch is *always* False: inv_freq is never recomputed and the update is a
    pure no-op whose only runtime effect is the illegal sync.

    Flipping rope_type to a static value skips that branch with bit-identical
    frequencies (inv_freq is untouched). Returns the number of modules neutralized.
    Only safe while the cache covers the whole decode range, so we require
    max_seq_len_cached >= max_target_positions before flipping.
    """
    max_target = getattr(getattr(model, "config", None), "max_target_positions", None)
    count = 0
    for module in model.modules():
        rope_type = getattr(module, "rope_type", None)
        if not (isinstance(rope_type, str) and "dynamic" in rope_type):
            continue
        cached = getattr(module, "max_seq_len_cached", None)
        if max_target is not None and cached is not None and cached < max_target:
            continue  # cache doesn't cover the decode range; leave dynamic behaviour
        module.rope_type = "default"
        count += 1
    return count


class _IncrementalMonotonicMask:
    """O(1)-per-step replacement for MonotonicTimeShiftLogitsProcessor.

    The stock processor re-scans the whole growing sequence every token (O(T^2)
    per window) to find the last TIME_SHIFT and whether it came after the last
    SOS. We track that as running per-row state instead: the mask forbids any
    time-shift token strictly smaller than the last emitted one (time can't go
    backwards), active only while a time-shift is the most recent structural
    event. Produces bit-identical masks to the stock processor (greedy-verified).
    """

    def __init__(self, tokenizer, device):
        self.ts_start = tokenizer.event_start[EventType.TIME_SHIFT]
        self.ts_end = tokenizer.event_end[EventType.TIME_SHIFT]
        self.sos_ids = torch.tensor(
            [tokenizer.sos_id] + list(getattr(tokenizer, "context_sos", {}).values()),
            dtype=torch.long, device=device)
        self.ts_vocab = torch.arange(self.ts_start, self.ts_end, device=device)
        self.last_ts_value = None  # (batch,) long
        self.active = None         # (batch,) bool

    def init_from_prompt(self, input_ids):
        device = input_ids.device
        b, L = input_ids.shape
        is_ts = (input_ids >= self.ts_start) & (input_ids < self.ts_end)
        is_sos = torch.isin(input_ids, self.sos_ids)
        idx = torch.arange(L, device=device).expand(b, -1)
        last_ts_idx = torch.max(torch.where(is_ts, idx, -1), dim=1).values
        last_sos_idx = torch.max(torch.where(is_sos, idx, -1), dim=1).values
        rows = torch.arange(b, device=device)
        self.last_ts_value = torch.where(
            last_ts_idx != -1,
            input_ids[rows, last_ts_idx.clamp(min=0)] - self.ts_start,
            torch.zeros(b, dtype=torch.long, device=device))
        self.active = (last_ts_idx != -1) & (last_ts_idx > last_sos_idx)
        return self

    def apply(self, scores):
        # Forbid time-shift *tokens* below the last emitted one, for active rows.
        # ts_vocab holds token ids, so the threshold is ts_start + last value (the
        # id of the last time-shift), matching the stock processor exactly.
        thresh = (self.ts_start + self.last_ts_value).unsqueeze(1)
        invalid = (self.ts_vocab.unsqueeze(0) < thresh) & self.active.unsqueeze(1)
        scores[:, self.ts_start:self.ts_end].masked_fill_(invalid, -float('inf'))
        return scores

    def update(self, tok):
        t = tok.reshape(-1)
        is_ts = (t >= self.ts_start) & (t < self.ts_end)
        is_sos = torch.isin(t, self.sos_ids)
        self.last_ts_value = torch.where(is_ts, t - self.ts_start, self.last_ts_value)
        self.active = torch.where(is_ts, torch.ones_like(self.active),
                                  torch.where(is_sos, torch.zeros_like(self.active), self.active))


class CUDAGraphDecoder:
    """Captures a fixed-shape single-token decode step into a CUDA graph.

    All inputs are static buffers; callers write new values then replay.
    The encoder hidden state and cache are also static (rewritten/reset externally).
    Captured once, reused across windows.
    """

    def __init__(self, model, batch_size, enc_shape, dtype):
        self.model = model
        self.batch_size = batch_size
        device = model.device
        self.vocab = model.config.vocab_size

        # Static input buffers
        self.static_token = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.static_cache_pos = torch.zeros((1,), dtype=torch.long, device=device)
        # encoder hidden state as a static buffer (rewritten per window)
        self.static_enc_hidden = torch.zeros(enc_shape, dtype=dtype, device=device)
        # static output
        self.static_logits = torch.zeros((batch_size, self.vocab),
                                         dtype=torch.float32, device=device)
        self._graph = None
        self._cache = None  # set at capture; reused (reset) across windows

    def capture(self, cache):
        """Warmup + capture the decode forward into a CUDA graph."""
        model = self.model
        self._cache = cache

        # Discover model_inputs structure with a probe call (outside graph)
        enc_out = BaseModelOutput(last_hidden_state=self.static_enc_hidden)
        probe = model.prepare_inputs_for_generation(
            self.static_token,
            past_key_values=cache,
            use_cache=True,
            encoder_outputs=enc_out,
            cache_position=self.static_cache_pos,
            decoder_attention_mask=None,
        )
        self._input_keys = [k for k, v in probe.items() if isinstance(v, torch.Tensor)]
        # Static input buffers matching probe shapes/dtypes
        self._static_inputs = {}
        for k in self._input_keys:
            t = probe[k]
            self._static_inputs[k] = torch.empty_like(t)
            self._static_inputs[k].copy_(t)
        self._extra_kwargs = {k: v for k, v in probe.items() if not isinstance(v, torch.Tensor)}

        def forward_only():
            outputs = model(**{**self._extra_kwargs, **self._static_inputs})
            self.static_logits.copy_(outputs.logits[:, -1, :].float())

        # Warmup on a side stream (required for cudagraph capture safety)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                forward_only()
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            forward_only()

    def set_encoder_hidden(self, enc_hidden):
        """Copy a new encoder hidden state into the static buffer (per window)."""
        self.static_enc_hidden.copy_(enc_hidden)

    def replay(self, token, cache_position):
        """Write the only two per-step inputs into the static buffers, replay.

        The graph probe (see ``capture``) showed only two tensor inputs vary
        across decode steps — ``decoder_input_ids`` (the last token) and
        ``cache_position``; everything else (encoder outputs, cache, attention
        mask, position ids) is static and baked into the graph. So we skip
        ``prepare_inputs_for_generation`` entirely in the hot loop and copy the
        two directly.
        """
        self._static_inputs["decoder_input_ids"].copy_(token)
        self._static_inputs["cache_position"].copy_(cache_position)
        self._graph.replay()
        return self.static_logits  # (B, vocab)


# Module-level cache of (decoder, cache) keyed by (model, batch_size, cfg, cache_len)
# so one graph is captured per cache-length bucket and reused across windows.
_decoder_cache = {}

# Decoder self-attention cache-length buckets. A static cache attends over its
# full allocated length every decode step, so we pick the smallest bucket that
# holds this window's prompt + generation headroom (a bigger cache = slower step).
# The last bucket is the model max.
_CACHE_BUCKETS = (384, 512, 768, 1024, 1280, 1536, 2048, 2560)
# Extra room reserved for generated tokens beyond the prompt when choosing a
# bucket. The prompt (lookback context) usually dominates the cache requirement;
# if a window generates past its bucket, model_generate_compiled retries it on a
# larger cache (see below), so this only needs to cover the common case.
_GEN_HEADROOM = 384


def _pick_bucket(prompt_len, model_max):
    """Smallest bucket >= prompt_len + headroom, capped at the model max."""
    need = min(prompt_len + _GEN_HEADROOM, model_max)
    for b in _CACHE_BUCKETS:
        if b >= need:
            return min(b, model_max)
    return model_max


def _bucket_ceil(need, model_max):
    """Smallest bucket >= need, capped at the model max (no generation headroom)."""
    for b in _CACHE_BUCKETS:
        if b >= need:
            return min(b, model_max)
    return model_max


def _get_decoder(model, batch_size, cfg_scale, enc_shape, dtype, cache_len):
    key = (id(model), batch_size, cfg_scale, cache_len)
    if key not in _decoder_cache:
        cache = get_cache(model, batch_size, 1, cfg_scale, max_cache_len=cache_len)
        decoder = CUDAGraphDecoder(model, batch_size, enc_shape, dtype)
        decoder.capture(cache)
        # After capture the cache holds garbage from the warmup forwards. The
        # decode loop calls _reset_cache + prefill before each window, which
        # restores correct state, so the capture corruption is harmless.
        _decoder_cache[key] = (decoder, cache)
    return _decoder_cache[key]


def _reset_cache(cache):
    """Reset the cache buffers for reuse (avoid realloc).

    Must go through EncoderDecoderCache.reset() (not the sub-caches directly) so
    the per-layer ``is_updated`` cross-attention flags are cleared too. Otherwise
    prefill sees ``is_updated=True`` (set during graph-capture warmup) and reuses
    the just-zeroed cross-attention K/V instead of recomputing it from the new
    window's encoder outputs — producing garbage logits.
    """
    cache.reset()


@torch.no_grad()
def model_generate_compiled(model, tokenizer, model_kwargs, generate_kwargs,
                            encoder_outputs: BaseModelOutput):
    """Custom decode loop with CUDA-graph-captured forward."""
    # Dynamic-RoPE models (e.g. v30) can't be graph-captured until their no-op
    # frequency update is disabled. Idempotent: a no-op once already flipped.
    neutralize_dynamic_rope(model)

    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in model_kwargs.items()}
    model_kwargs = {k: v.to(model.dtype) if k != "inputs" and isinstance(v, torch.Tensor)
                    and v.dtype == torch.float32 else v for k, v in model_kwargs.items()}

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
    do_sample = generate_kwargs.pop('do_sample', True)
    top_p = generate_kwargs.pop('top_p', 0.95)
    top_k = generate_kwargs.pop('top_k', 0)
    max_length = generate_kwargs.pop('max_length', 2560)
    if context_type is not None:
        context_type = ContextType(context_type)

    decoder_input_ids = model_kwargs["decoder_input_ids"]
    decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
    negative_prompt = model_kwargs.get("negative_prompt")
    negative_prompt_attention_mask = model_kwargs.get("negative_prompt_attention_mask")
    pad_token_id = getattr(tokenizer, 'pad_id', None)
    device = model.device

    eff_batch = decoder_input_ids.shape[0] * (2 if (cfg_scale > 1 and negative_prompt is not None) else 1)
    eos_token_ids = get_eos_token_id(tokenizer, lookback_time=lookback_time,
                                     lookahead_time=lookahead_time, context_type=context_type)
    logits_processor_list = build_logits_processors(
        tokenizer, cfg_scale, timeshift_bias, types_first, temperature,
        timing_temperature, mania_column_temperature, taiko_hit_temperature,
        lookback_time, device,
    )
    # Hoist MonotonicTimeShift out of the per-step list into an O(1) incremental
    # masker. Only safe without CFG (CFG merges the 2-stream logits before the
    # monotonic mask and the merged-batch state bookkeeping differs); with CFG we
    # keep the stock O(T^2) processor.
    mono_mask = None
    if cfg_scale <= 1:
        rest = LogitsProcessorList([p for p in logits_processor_list
                                    if not isinstance(p, MonotonicTimeShiftLogitsProcessor)])
        if len(rest) != len(logits_processor_list):  # one was actually removed
            mono_mask = _IncrementalMonotonicMask(tokenizer, device)
            logits_processor_list = rest

    prompt_len = decoder_input_ids.shape[1]
    b0 = decoder_input_ids.shape[0]
    model_max = model.config.max_target_positions
    enc_hidden = encoder_outputs.last_hidden_state  # (B, L_enc, D)
    eos_t = torch.tensor(eos_token_ids, device=device, dtype=torch.long)
    # The true generation limit, matching the non-compiled path (which caps at
    # max_length and its full-size cache). We must never stop a window short of
    # this just because a bucket was too small.
    hard_cap = min(max_length, model_max)

    def _run_window(cache_len):
        """Prefill + captured decode with a given static-cache length.

        Returns (generated_tokens, truncated) where ``truncated`` means the loop
        stopped because it filled the cache (``win_max``) rather than emitting EOS.
        A static cache attends over its full length every step, so a right-sized
        cache is cheaper; we start small and only grow on truncation (below).
        """
        win_max = min(hard_cap, cache_len)
        decoder, cache = _get_decoder(model, eff_batch, cfg_scale, enc_hidden.shape,
                                      model.dtype, cache_len)
        _reset_cache(cache)                    # fresh generation for this window
        decoder.set_encoder_hidden(enc_hidden)  # write this window's encoder state
        if mono_mask is not None:
            mono_mask.init_from_prompt(decoder_input_ids)  # reset per (re)attempt

        # 1. PREFILL (eager) — process the whole prompt, fill the cache. Must use
        # the SAME cache object the graph references so decode steps see it.
        cache_position = torch.arange(prompt_len, device=device)
        prefill_inputs = model.prepare_inputs_for_generation(
            decoder_input_ids,
            past_key_values=cache,
            use_cache=True,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            negative_prompt=negative_prompt,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            cache_position=cache_position,
        )
        prefill_out = model(**prefill_inputs)
        last_logits = prefill_out.logits[:, -1, :].float()  # (B, vocab)

        # Preallocated id buffer: O(1) append + O(1) view of the sequence so far,
        # replacing the per-step growing torch.cat (O(T^2)/window). Processors that
        # still need history (types_first) read a cheap slice view.
        id_buffer = torch.empty((b0, win_max), dtype=torch.long, device=device)
        id_buffer[:, :prompt_len] = decoder_input_ids

        generated = []
        cur_len = prompt_len
        truncated = True
        # 2. DECODE LOOP — `last_logits` holds the logits for the current position
        # (prefill output on entry, else the previous replay). Sample, check EOS,
        # then replay for the next position. The EOS token is kept (appended) so
        # downstream trimming matches HF.
        while cur_len < win_max:
            if mono_mask is not None:
                last_logits = mono_mask.apply(last_logits)
            proc_logits = logits_processor_list(id_buffer[:, :cur_len], last_logits)
            # Temperature is already applied by logits_processor_list, so _sample
            # must NOT re-apply it — pass 1.0. It still handles top-p / top-k, which
            # the processor list omits (matching model.generate's sampling step).
            next_token = _sample(proc_logits, do_sample, top_p, top_k, 1.0)

            id_buffer[:, cur_len] = next_token.reshape(b0)
            generated.append(next_token)
            if mono_mask is not None:
                mono_mask.update(next_token)
            cur_len += 1

            if torch.isin(next_token, eos_t).any().item():
                truncated = False
                break
            cache_position = torch.tensor([cur_len - 1], device=device, dtype=torch.long)
            last_logits = decoder.replay(next_token, cache_position)
        return generated, truncated

    start_time = time.perf_counter()
    # Start with a small bucket sized for the common short-generation case.
    cache_len = _pick_bucket(prompt_len, model_max)
    generated, truncated = _run_window(cache_len)
    # If the window filled its bucket without hitting EOS while the model was still
    # allowed to generate (bucket < hard_cap), the bucket was too small and we'd
    # otherwise emit a truncated sequence (seen with small lookahead, where windows
    # generate far more than the default headroom). Retry once with a cache big
    # enough for the full budget so compiled output matches the non-compiled path.
    if truncated and cache_len < hard_cap:
        cache_len = _bucket_ceil(hard_cap, model_max)
        generated, truncated = _run_window(cache_len)
    elapsed = time.perf_counter() - start_time

    gen_tensor = torch.cat(generated, dim=1) if generated else torch.empty(
        (decoder_input_ids.shape[0], 0), dtype=torch.long, device=device)
    result = torch.cat([decoder_input_ids, gen_tensor], dim=1).cpu()
    stats = _build_generation_stats(result, model_kwargs, pad_token_id, elapsed)
    return result, stats


def _sample(logits, do_sample, top_p, top_k, temperature):
    """Top-p (nucleus) sampling. Returns (B, 1) token tensor."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, top_k)
        thresh = vals[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, -float('inf')), logits)
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float('inf'))
        logits = torch.full_like(logits, -float('inf'))
        logits.scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    flat = probs.view(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1)
    return sampled.view(probs.size(0), -1)
