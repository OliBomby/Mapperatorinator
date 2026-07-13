import _pickle
import os
import time
import threading
import traceback
import torch
from multiprocessing.connection import Listener, Client

from transformers import LogitsProcessorList, ClassifierFreeGuidanceLogitsProcessor, TemperatureLogitsWarper
from transformers.modeling_outputs import BaseModelOutput

from ..event import EventType, ContextType
from .logit_processors import ConditionalTemperatureLogitsWarper, get_beat_type_tokens, \
    get_mania_type_tokens, get_scroll_speed_tokens, TimeshiftBias, LookbackBiasLogitsWarper, \
    MonotonicTimeShiftLogitsProcessor
from .cache_utils import get_cache
from ..model import Mapperatorinator
from ..tokenizer import Tokenizer

# The default address used for IPC
SOCKET_PATH = r'\\.\pipe\Mapperatorinator'

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10

RETRY_SIGNAL = "RETRY_SIGNAL"

# generate_kwargs['op'] value that requests encoder precompute instead of generation
PRECOMPUTE_ENCODER_OP = "precompute_encoder"


def _prompt_token_counts(model_kwargs, pad_token_id: int | None) -> torch.Tensor | None:
    decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
    if isinstance(decoder_attention_mask, torch.Tensor):
        return decoder_attention_mask.to(torch.long).sum(dim=-1).cpu()

    decoder_input_ids = model_kwargs.get("decoder_input_ids")
    if not isinstance(decoder_input_ids, torch.Tensor):
        return None

    if pad_token_id is None:
        return torch.full((decoder_input_ids.shape[0],), decoder_input_ids.shape[1], dtype=torch.long)

    return decoder_input_ids.ne(pad_token_id).to(torch.long).sum(dim=-1).cpu()


def _output_token_counts(result: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    if pad_token_id is None:
        return torch.full((result.shape[0],), result.shape[1], dtype=torch.long)

    return result.ne(pad_token_id).to(torch.long).sum(dim=-1)


def _build_generation_stats(
        result: torch.Tensor,
        model_kwargs: dict,
        pad_token_id: int | None,
        elapsed_seconds: float,
) -> dict:
    prompt_token_counts = _prompt_token_counts(model_kwargs, pad_token_id)
    generated_token_counts = _output_token_counts(result, pad_token_id)
    if prompt_token_counts is not None:
        generated_token_counts = torch.clamp(generated_token_counts - prompt_token_counts, min=0)

    generated_tokens = int(generated_token_counts.sum().item())
    tokens_per_second = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return {
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sample": generated_token_counts.tolist(),
        "elapsed_seconds": float(elapsed_seconds),
        "tokens_per_second": tokens_per_second,
    }


def get_eos_token_id(tokenizer, lookback_time: float = 0, lookahead_time: float = 0, context_type: ContextType = None):
    eos_token_id = [tokenizer.eos_id]
    if context_type is not None and context_type in tokenizer.context_eos:
        eos_token_id.append(tokenizer.context_eos[context_type])
    if lookback_time > 0:
        eos_token_id.extend(range(tokenizer.event_start[EventType.TIME_SHIFT], tokenizer.event_start[EventType.TIME_SHIFT] + int(lookback_time / MILISECONDS_PER_STEP)))
    if lookahead_time > 0:
        eos_token_id.extend(range(tokenizer.event_end[EventType.TIME_SHIFT] - int(lookahead_time / MILISECONDS_PER_STEP), tokenizer.event_end[EventType.TIME_SHIFT]))
    return eos_token_id


def build_logits_processors(tokenizer, cfg_scale, timeshift_bias, types_first,
                            temperature, timing_temperature, mania_column_temperature,
                            taiko_hit_temperature, lookback_time, device):
    """Create the logits processors used by every generate path."""
    logits_processor_list = LogitsProcessorList()
    if cfg_scale > 1.0:
        logits_processor_list.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))

    logits_processor_list.append(MonotonicTimeShiftLogitsProcessor(tokenizer))

    if timeshift_bias != 0:
        logits_processor_list.append(
            TimeshiftBias(
                timeshift_bias,
                tokenizer.event_start[EventType.TIME_SHIFT],
                tokenizer.event_end[EventType.TIME_SHIFT]
            )
        )
    if types_first:
        logits_processor_list.append(ConditionalTemperatureLogitsWarper(
            temperature,
            timing_temperature,
            mania_column_temperature,
            taiko_hit_temperature,
            types_first,
            get_beat_type_tokens(tokenizer),
            get_mania_type_tokens(tokenizer),
            get_scroll_speed_tokens(tokenizer),
        ))
    else:
        logits_processor_list.append(TemperatureLogitsWarper(temperature))
    if lookback_time > 0:
        logits_processor_list.append(LookbackBiasLogitsWarper(lookback_time, tokenizer, types_first, device))
    return logits_processor_list


@torch.no_grad()
def precompute_encoder_outputs(model, frames: torch.Tensor, cond_kwargs: dict,
                               song_positions: torch.Tensor | None = None,
                               batch_size: int = 16) -> torch.Tensor:
    """Run the encoder (mel + transformer encoder + conditioning) for all windows.

    The encoder is a pure function of the audio window + static conditioning, so
    computing all windows in large batches before the sequential decode loop is
    much cheaper than one window at a time inside it.

    Args:
        model: Mapperatorinator
        frames: (N, L_raw) raw audio windows, fp32
        cond_kwargs: dict of beatmap_idx/difficulty/mapper_idx tensors, either
                     scalar (broadcast across the N windows) or per-window (N,)
        song_positions: (N, 2) tensor of (pos_start, pos_end) per window, or None
        batch_size: chunk size for the encoder forward

    Returns: (N, L_enc, D) tensor of encoder hidden states in input order, on
             CPU so long songs don't pin the whole song's encoder state in VRAM
             (a one-hour song is thousands of windows / several GB). Callers
             move one window at a time back to the device, which is negligible
             (~1.5 MB per window) next to the decode cost.
    """
    n = frames.shape[0]
    device = model.device
    chunks = []
    model.eval()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = frames[start:end].to(device)  # (b, L_raw)
        bsz = chunk.shape[0]

        def rows(v):
            # Conditioning for this chunk: broadcast scalars, slice per-window tensors
            if v is None:
                return None
            v = v.to(device)
            return v.expand(bsz) if v.numel() == 1 else v[start:end]

        # NOTE on determinism: batched matmuls/attention use different GPU kernels
        # than single-row, so the encoder output is NOT bit-identical to the
        # original one-window-at-a-time path. This propagates into sampling
        # divergence, so output is quality-equivalent but not byte-identical
        # (greedy still matches closely; only sampled runs diverge). Comparable to
        # bf16-vs-fp32 drift.
        mel = model.spectrogram(chunk).to(model.transformer.dtype)  # (b, L_mel, M)

        conds = []
        if model.do_style_embed:
            beatmap_idx = rows(cond_kwargs.get("beatmap_idx"))
            if beatmap_idx is None:
                beatmap_idx = torch.full((bsz,), model.num_classes, dtype=torch.long, device=device)
            conds.append(model.style_embedder(beatmap_idx))
        if model.do_difficulty_embed:
            difficulty = rows(cond_kwargs.get("difficulty"))
            if difficulty is None:
                difficulty = torch.full((bsz,), 5.0, device=device)
            # Cast to the model dtype so the RBF basis matches difficulty_proj's
            # weights. The proj is a bf16 nn.Linear when the model is bf16, and the
            # basis buffers are cast with it; an fp32 difficulty would make the
            # basis fp32 and trip "mat1 and mat2 must have the same dtype".
            difficulty = difficulty.to(model.transformer.dtype)
            conds.append(model.difficulty_embedder(difficulty))
        if model.do_mapper_embed:
            mapper_idx = rows(cond_kwargs.get("mapper_idx"))
            if mapper_idx is None:
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
        chunks.append(enc_out.last_hidden_state.cpu())  # (b, L_enc, D)

    return torch.cat(chunks, dim=0)


@torch.no_grad()
def model_precompute_encoder(model, model_kwargs):
    """Server-side handler for the precompute-encoder op. Returns ((N, L_enc, D)
    CPU tensor, stats)."""
    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}
    frames = model_kwargs['inputs']
    cond_kwargs = {k: model_kwargs[k] for k in ("beatmap_idx", "difficulty", "mapper_idx") if model_kwargs.get(k) is not None}
    song_positions = model_kwargs.get('song_position')

    start_time = time.perf_counter()
    hidden = precompute_encoder_outputs(model, frames, cond_kwargs, song_positions,
                                        batch_size=frames.shape[0])  # returned on CPU
    elapsed_seconds = time.perf_counter() - start_time

    stats = {
        "generated_tokens": 0,
        "generated_tokens_per_sample": [0] * hidden.shape[0],
        "elapsed_seconds": float(elapsed_seconds),
        "tokens_per_second": 0.0,
    }
    return hidden, stats


@torch.no_grad()
def model_generate(model, tokenizer, model_kwargs, generate_kwargs):
    # To device
    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}
    model_kwargs = {k: v.to(model.dtype) if k != "inputs" and isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v for k, v in model_kwargs.items()}

    # Precomputed encoder hidden states (see precompute_encoder_outputs) replace
    # the raw audio: model.generate skips the encoder prefill entirely.
    encoder_hidden = model_kwargs.pop('encoder_outputs', None)
    if encoder_hidden is not None:
        model_kwargs.pop('inputs', None)
        batch_size = encoder_hidden.shape[0]
        model_kwargs['encoder_outputs'] = BaseModelOutput(last_hidden_state=encoder_hidden)
    else:
        batch_size = model_kwargs['inputs'].shape[0]
    # print(f"[Model Generate] Batch size: {batch_size}, Model device: {model.device}")

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
        context_type = ContextType(context_type)  # Convert to ContextType enum

    # Create the logits processors
    logits_processor_list = build_logits_processors(
        tokenizer, cfg_scale, timeshift_bias, types_first, temperature,
        timing_temperature, mania_column_temperature, taiko_hit_temperature,
        lookback_time, model.device,
    )

    # Prepare cache
    cache = get_cache(model, batch_size, generate_kwargs.get('num_beams', 1), cfg_scale)
    pad_token_id = generate_kwargs.get('pad_token_id', getattr(tokenizer, 'pad_id', None))

    # Perform batched generation
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=precision == 'amp'):
        start_time = time.perf_counter()
        result = model.generate(
            **model_kwargs,
            **generate_kwargs,
            use_cache=True,
            past_key_values=cache,
            logits_processor=logits_processor_list,
            eos_token_id=get_eos_token_id(tokenizer, lookback_time=lookback_time, lookahead_time=lookahead_time, context_type=context_type),
        )
        elapsed_seconds = time.perf_counter() - start_time

    result = result.cpu()
    stats = _build_generation_stats(result, model_kwargs, pad_token_id, elapsed_seconds)

    return result, stats


@torch.no_grad()
def model_forward(model, model_kwargs, generate_kwargs):
    # To device
    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}
    model_kwargs = {k: v.to(model.dtype) if k != "inputs" and isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v for k, v in model_kwargs.items()}
    model_kwargs["frames"] = model_kwargs.pop('inputs', None)  # Rename for compatibility
    precision = generate_kwargs.pop('precision', 'fp32')
    cfg_scale = generate_kwargs.pop('cfg_scale', 1.0)

    # Prepare inputs for the model
    model_kwargs = model.prepare_inputs_for_generation(**model_kwargs)

    # Create the logits processors
    logits_processor_list = LogitsProcessorList()
    if cfg_scale > 1.0:
        logits_processor_list.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))

    # Perform forward pass
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=precision == 'amp'):
        logits = model.forward(**model_kwargs).logits.to(torch.float32)

    logits = logits_processor_list(model_kwargs["decoder_input_ids"], logits).cpu()
    return logits


if os.name == "nt":
    import msvcrt

    def portable_lock(fp):
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)

    def portable_unlock(fp):
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def portable_lock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def portable_unlock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class Locker:
    def __enter__(self):
        self.fp = open("./lockfile.lck", 'w+')
        portable_lock(self.fp)

    def __exit__(self, _type, value, tb):
        portable_unlock(self.fp)
        self.fp.close()



class InferenceServer:
    def __init__(
            self,
            model,
            tokenizer,
            max_batch_size=8,
            batch_timeout=0.2,
            idle_timeout=20,
            socket_path=SOCKET_PATH,
            fast_decoder_loop=False,
    ):
        """
        Initializes the inference server.
        :param model: The model to use for inference.
        :param tokenizer: The tokenizer to use for processing inputs.
        :param max_batch_size: Maximum batch size for processing requests.
        :param batch_timeout: Time in seconds to wait for more requests before processing a batch.
        :param idle_timeout: Time in seconds to wait before shutting down due to no clients.
        :param socket_path: The address used for IPC.
        :param fast_decoder_loop: Use the CUDA-graph fast decoder loop for batch-1
            requests with precomputed encoder outputs (see compiled_decode.py).
        """
        self.model: Mapperatorinator = model
        self.tokenizer: Tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.idle_timeout = idle_timeout
        self.socket_path = socket_path
        self.fast_decoder_loop = fast_decoder_loop
        self.grouped_requests = {}  # holds pending requests
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.listener = None
        self.connections = 0
        self.request_event = threading.Event()  # set when a request is enqueued

    def start(self):
        # Remove stale socket
        try:
            os.unlink(self.socket_path)
        except (FileNotFoundError, OSError):
            pass

        # Start IPC listener
        self.listener = Listener(self.socket_path)
        threading.Thread(target=self._listener_thread, daemon=True).start()
        # Start batcher thread
        threading.Thread(target=self._batch_thread, daemon=True).start()
        # Start idle monitor
        threading.Thread(target=self._idle_monitor, daemon=True).start()

    def stop(self):
        self.shutdown_flag.set()
        try:
            if self.listener is not None:
                self.listener.close()
        except Exception:
            pass
        try:
            os.unlink(self.socket_path)
        except Exception:
            pass

    def _listener_thread(self):
        while not self.shutdown_flag.is_set():
            try:
                conn = self.listener.accept()
                # Handle each client in its own thread
                threading.Thread(target=self._client_handler, args=(conn,), daemon=True).start()
            except (OSError, EOFError) as e:
                print(f"[Listener] Error in accept: {e}")
                time.sleep(1)  # Wait before retrying

    def _client_handler(self, conn):
        with self.lock:
            self.connections += 1
        try:
            with conn:
                while True:
                    try:
                        model_kwargs, generate_kwargs = conn.recv()
                    except _pickle.UnpicklingError:
                        print("UnpicklingError detected! Requesting a retry from the client.")
                        # Tell the client to try again
                        conn.send(RETRY_SIGNAL)
                        # Loop back to conn.recv() to wait for the resent data
                        continue
                    except (EOFError, OSError):
                        break

                    # Group by generate_kwargs AND input kind: requests carrying
                    # precomputed encoder outputs cannot be collated with requests
                    # carrying raw audio (different tensor keys).
                    generate_kwargs_set = (frozenset(generate_kwargs.items()), 'encoder_outputs' in model_kwargs)

                    # Prepare a response event
                    response_event = threading.Event()
                    inputs = model_kwargs.get('inputs')
                    batch_size = inputs.shape[0] if inputs is not None else model_kwargs['encoder_outputs'].shape[0]
                    record = {
                        'model_kwargs': model_kwargs,
                        'total_work': batch_size,
                        'work_done': 0,
                        'conn': conn,
                        'event': response_event,
                        'result': None,
                        'generated_tokens': 0,
                        'elapsed_seconds': 0.0,
                    }

                    # Enqueue request
                    with self.lock:
                        if generate_kwargs_set in self.grouped_requests:
                            self.grouped_requests[generate_kwargs_set].append(record)
                        else:
                            self.grouped_requests[generate_kwargs_set] = [record]
                        self.request_event.set()

                    # Wait until batch thread processes it
                    response_event.wait()

                    # Send back result
                    try:
                        conn.send(record['result'])
                    except BrokenPipeError:
                        # Client disconnected
                        break
        finally:  # Ensure we always close the connection
            with self.lock:
                self.connections -= 1

    def _batch_thread(self):
        while not self.shutdown_flag.is_set():
            # Wake as soon as a request is enqueued instead of polling on a fixed
            # sleep (which taxed every request of a sequential client with up to
            # batch_timeout of latency).
            self.request_event.wait(timeout=self.batch_timeout)
            with self.lock:
                if not self.grouped_requests:
                    # Enqueue + set happen under this lock, so clearing here
                    # cannot drop a wakeup.
                    self.request_event.clear()
                    continue
                gather = self.connections > 1
            if gather:
                # Multiple clients are connected: wait a gather window so their
                # concurrent requests can merge into one batch. A single client
                # is necessarily sequential, so gathering would only add latency.
                time.sleep(self.batch_timeout)
            with self.lock:
                if not self.grouped_requests:
                    continue
                generate_kwargs_set = list(self.grouped_requests.keys())[0]
                requests: list = self.grouped_requests[generate_kwargs_set]

                generate_kwargs: dict = dict(generate_kwargs_set[0])
                cfg_scale = generate_kwargs.get('cfg_scale', 1.0)
                num_beams = generate_kwargs.get('num_beams', 1)
                batch_multiplier = 2 * num_beams if cfg_scale > 1 else num_beams

                # Grab full or partial requests until BATCH_SIZE is reached or requests is empty
                batch_requests = []
                remaining_batch_size = self.max_batch_size // batch_multiplier
                while remaining_batch_size > 0 and len(requests) > 0:
                    request = requests.pop(0)
                    req_kwargs = request['model_kwargs']
                    req_total_work = request['total_work']
                    req_work_done = request['work_done']
                    req_remaining_work = req_total_work - req_work_done
                    work = min(req_remaining_work, remaining_batch_size)
                    batch_requests.append((self._cut_model_kwargs(req_kwargs, req_work_done, work), request, work))
                    remaining_batch_size -= work
                    if req_remaining_work > work:
                        # If there is still work left, re-add the record to the queue
                        requests.insert(0, request)

                if not self.grouped_requests[generate_kwargs_set]:
                    del self.grouped_requests[generate_kwargs_set]

            try:
                # Collate inputs
                keys = [k for k in batch_requests[0][0].keys() if batch_requests[0][0][k] is not None]
                model_kwargs = {}
                paddings = [0 for _ in range(len(batch_requests))]  # For padding left
                for k in keys:
                    kwargses = [b[0][k] for b in batch_requests]
                    # Pad left if necessary
                    if kwargses[0].dim() > 1:
                        max_len = max(tensor.size(-1) for tensor in kwargses)
                        if k == 'decoder_input_ids':
                            paddings = [max_len - tensor.size(-1) for tensor in kwargses]
                        kwargses = [torch.nn.functional.pad(tensor, (max_len - tensor.size(-1), 0)) for tensor in kwargses]
                    model_kwargs[k] = torch.cat(kwargses, dim=0)

                op = generate_kwargs.pop('op', None)
                if op == PRECOMPUTE_ENCODER_OP:
                    outputs, stats = model_precompute_encoder(self.model, model_kwargs)
                else:
                    outputs, stats = self._model_generate(model_kwargs, generate_kwargs)
                generated_tokens_per_sample = stats.get('generated_tokens_per_sample', [])

                # Split and dispatch results
                batch_i = 0
                for i, (_, request, work_done) in enumerate(batch_requests):
                    padding = paddings[i]
                    out = outputs[batch_i:batch_i + work_done, padding:]  # Remove padding from the left
                    request_generated_tokens = sum(generated_tokens_per_sample[batch_i:batch_i + work_done])
                    batch_i += work_done
                    request['result'] = out if request['result'] is None else torch.cat((request['result'], out), dim=0)
                    request['work_done'] += work_done
                    request['generated_tokens'] += request_generated_tokens
                    request['elapsed_seconds'] += stats.get('elapsed_seconds', 0.0)
                    if request['work_done'] >= request['total_work']:
                        # All work done for this record, signal completion
                        elapsed_seconds = request['elapsed_seconds']
                        generated_tokens = request['generated_tokens']
                        request['result'] = {
                            'output': request['result'],
                            'stats': {
                                'generated_tokens': generated_tokens,
                                'elapsed_seconds': elapsed_seconds,
                                'tokens_per_second': generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                            },
                        }
                        request['event'].set()
            except Exception as e:
                print(f"[Batch Thread] Error processing batch: {e}")
                traceback.print_exc()
                # Signal all requests in this batch to retry
                for _, request, _ in batch_requests:
                    request['result'] = RETRY_SIGNAL
                    request['event'].set()  # Signal completion
            finally:
                torch.cuda.empty_cache()  # Clear any cached memory, otherwise will definitely run out of memory if multiple batch sizes are used

    def _model_generate(self, model_kwargs, generate_kwargs):
        # Batch-1 requests with precomputed encoder outputs go through the fast
        # decoder loop when enabled; merged batches and beam search use the stock
        # generate (the fast loop is captured for a fixed batch size and stops on
        # the first EOS, which is wrong for independent batched sequences).
        if (self.fast_decoder_loop
                and isinstance(model_kwargs.get('encoder_outputs'), torch.Tensor)
                and model_kwargs['encoder_outputs'].shape[0] == 1
                and generate_kwargs.get('num_beams', 1) == 1):
            from .compiled_decode import model_generate_compiled
            return model_generate_compiled(self.model, self.tokenizer, model_kwargs, generate_kwargs)
        return model_generate(self.model, self.tokenizer, model_kwargs, generate_kwargs)

    def _cut_model_kwargs(self, model_kwargs, start, length):
        """Cuts the model_kwargs tensors to the specified range."""
        return {k: v[start:start + length] if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}

    def _idle_monitor(self):
        last_activity = time.time()
        while not self.shutdown_flag.is_set():
            time.sleep(self.idle_timeout / 2)
            with self.lock:
                if self.connections > 0:
                    last_activity = time.time()
            if time.time() - last_activity > self.idle_timeout:
                # No requests for a while: shutdown
                self.shutdown_flag.set()
                try:
                    self.listener.close()
                    os.unlink(self.socket_path)
                except Exception:
                    pass


class InferenceClient:
    def __init__(
            self,
            model_loader,
            tokenizer_loader,
            max_batch_size=8,
            batch_timeout=0.2,
            idle_timeout=20,
            server_thread_daemon=False,
            socket_path=SOCKET_PATH,
            fast_decoder_loop=False,
    ):
        """
        Initializes the inference client. Automatically starts the inference server if it is not running.
        :param model_loader: Function to load the model.
        :param tokenizer_loader: Function to load the tokenizer.
        :param max_batch_size: Maximum batch size for processing requests.
        :param batch_timeout: Time in seconds to wait for more requests before processing a batch.
        :param idle_timeout: Time in seconds to wait before shutting down due to no clients.
        :param server_thread_daemon: Whether the auto-started background server thread should be daemonized.
        :param socket_path: The address used for IPC.
        :param fast_decoder_loop: Enable the CUDA-graph fast decoder loop on the auto-started server.
        """
        self.model_loader = model_loader
        self.tokenizer_loader = tokenizer_loader
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.idle_timeout = idle_timeout
        self.server_thread_daemon = server_thread_daemon
        self.socket_path = socket_path
        self.fast_decoder_loop = fast_decoder_loop
        self.conn = None
        self.last_generation_stats = None
        self._server = None
        self._server_thread = None

    def __enter__(self):
        self._reconnect()
        return self

    def _reconnect(self):
        with Locker():
            try:
                self.conn = Client(self.socket_path)
            except FileNotFoundError:
                # No server: start one
                self._start_server_thread()
                # Wait for server socket to appear
                while not os.path.exists(self.socket_path):
                    time.sleep(0.1)
                self.conn = Client(self.socket_path)

    def _start_server_thread(self):
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        self._server_thread = threading.Thread(
            target=self._start_server,
            args=(self.model_loader, self.tokenizer_loader),
            daemon=self.server_thread_daemon,
        )
        self._server_thread.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.conn:
            self.conn.close()

    def _start_server(self, model_loader, tokenizer_loader):
        # Load model inside server process
        model = model_loader()
        tokenizer = tokenizer_loader()
        server = InferenceServer(
            model,
            tokenizer,
            max_batch_size=self.max_batch_size,
            batch_timeout=self.batch_timeout,
            idle_timeout=self.idle_timeout,
            socket_path=self.socket_path,
            fast_decoder_loop=self.fast_decoder_loop,
        )
        self._server = server
        server.start()
        # Block until shutdown
        while not server.shutdown_flag.is_set():
            time.sleep(1)
        self._server = None

    def generate(self, model_kwargs, generate_kwargs, max_retries=3):
        attempts = 0
        while attempts < max_retries:
            # Send request and wait for response
            try:
                self.conn.send((model_kwargs, generate_kwargs))
                result = self.conn.recv()
            except (EOFError, OSError):
                print("Connection error, attempting to reconnect...")
                self._reconnect()
                attempts += 1
                continue

            if result == RETRY_SIGNAL:
                print("Retrying request due to Error.")
                attempts += 1
                continue
            else:
                if isinstance(result, dict) and 'output' in result:
                    self.last_generation_stats = result.get('stats')
                    return result['output']
                self.last_generation_stats = None
                return result

        raise RuntimeError(f"Failed to get a valid response after {max_retries} attempts.")

    def precompute_encoder(self, model_kwargs):
        """Precompute encoder outputs for N windows on the server.

        model_kwargs must hold 'inputs' (N, L_raw) plus optional per-window
        conditioning ('beatmap_idx'/'difficulty'/'mapper_idx' as (N,) tensors,
        'song_position' as (N, 2)). Returns an (N, L_enc, D) CPU tensor to pass
        per window as model_kwargs['encoder_outputs'] in generate requests.
        """
        return self.generate(model_kwargs, {'op': PRECOMPUTE_ENCODER_OP})

    def ensure_server(self):
        """Ensure the background inference server is running.

        This is useful when an external owner (e.g., a web UI) wants to start the
        server once and keep it alive independently of any per-job client
        connections.
        """
        with Locker():
            if not os.path.exists(self.socket_path):
                self._start_server_thread()

        # Wait for server socket to appear.
        while not os.path.exists(self.socket_path):
            time.sleep(0.1)

    def shutdown_server(self, join_timeout=5.0):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            finally:
                self.conn = None

        if self._server is not None:
            self._server.stop()

        server_thread = self._server_thread
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=join_timeout)


if __name__ == "__main__":
    ckpt_path_str = "OliBomby/Mapperatorinator-v32"

    # Example usage
    def model_loader():
        model = Mapperatorinator.from_pretrained(ckpt_path_str)
        model.generation_config.disable_compile = True
        model.eval()
        model.to('cuda')
        return model

    def tokenizer_loader():
        return Tokenizer.from_pretrained(ckpt_path_str)

    client = InferenceClient(model_loader, tokenizer_loader)
    tokenizer = Tokenizer.from_pretrained(ckpt_path_str)

    # Example model_kwargs and generate_kwargs
    model_kwargs = {
        'inputs': torch.rand((1, 524160)),  # Example input
        'difficulty': torch.tensor([7.]),
        'mapper_idx': torch.tensor([-1]),
        'song_position': torch.tensor([[0., .112]]),
    }
    generate_kwargs = {
        'num_beams': 1,
        'max_length': 2048,
        'do_sample': True,
        'cfg_scale': 1.0,
        'top_p': 0.9,
        'top_k': 0,
        'pad_token_id': tokenizer.pad_id,
        'timeshift_bias': 0,
        'types_first': False,
        'temperature': 0.9,
        'timing_temperature': 0.0,
        'mania_column_temperature': 0.7,
        'taiko_hit_temperature': 0.7,
        'lookback_time': 0,
        'lookahead_time': 3000,
    }

    result = client.generate(model_kwargs, generate_kwargs)
    events = [tokenizer.decode(t) if t > 10 else t for t in result[0].numpy()]
    print(events)  # Process the result as needed
    print(client.last_generation_stats)
