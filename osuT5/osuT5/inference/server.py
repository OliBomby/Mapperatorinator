import os
import time
import threading
import torch
from multiprocessing.connection import Listener, Client

from transformers import LogitsProcessorList, ClassifierFreeGuidanceLogitsProcessor, TemperatureLogitsWarper

from ..event import EventType
from .logit_processors import ConditionalTemperatureLogitsWarper, get_beat_type_tokens, \
    get_mania_type_tokens, get_scroll_speed_tokens, TimeshiftBias, LookbackBiasLogitsWarper
from .cache_utils import get_cache
from ..model import Mapperatorinator
from ..tokenizer import Tokenizer

# The default address used for IPC
SOCKET_PATH = r'\\.\pipe\Mapperatorinator'

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


def get_eos_token_id(tokenizer, lookback_time: float = 0, lookahead_time: float = 0):
    eos_token_id = [tokenizer.eos_id]
    eos_token_id.extend(tokenizer.context_eos.values())
    if lookback_time > 0:
        eos_token_id.extend(range(tokenizer.event_start[EventType.TIME_SHIFT], tokenizer.event_start[EventType.TIME_SHIFT] + int(lookback_time / MILISECONDS_PER_STEP)))
    if lookahead_time > 0:
        eos_token_id.extend(range(tokenizer.event_end[EventType.TIME_SHIFT] - int(lookahead_time / MILISECONDS_PER_STEP), tokenizer.event_end[EventType.TIME_SHIFT]))
    return eos_token_id


def model_generate(model, tokenizer, model_kwargs, generate_kwargs):
    # To device
    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}
    batch_size = model_kwargs['inputs'].shape[0]

    cfg_scale = generate_kwargs.pop('cfg_scale', 1.0)
    timeshift_bias = generate_kwargs.pop('timeshift_bias', 0)
    types_first = generate_kwargs.pop('types_first', False)
    temperature = generate_kwargs.pop('temperature', 1.0)
    timing_temperature = generate_kwargs.pop('timing_temperature', temperature)
    mania_column_temperature = generate_kwargs.pop('mania_column_temperature', temperature)
    taiko_hit_temperature = generate_kwargs.pop('taiko_hit_temperature', temperature)
    lookback_time = generate_kwargs.pop('lookback_time', 0.0)
    lookahead_time = generate_kwargs.pop('lookahead_time', 0.0)

    # Create the logits processors
    logits_processor_list = LogitsProcessorList()
    if cfg_scale > 1.0:
        logits_processor_list.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))
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
        logits_processor_list.append(LookbackBiasLogitsWarper(lookback_time, tokenizer, types_first, model.device))

    # Prepare cache
    cache = get_cache(model, batch_size, generate_kwargs.get('num_beams', 1), cfg_scale)

    # Perform batched generation
    return model.generate(
        **model_kwargs,
        **generate_kwargs,
        use_cache=True,
        past_key_values=cache,
        logits_processor=logits_processor_list,
        eos_token_id=get_eos_token_id(tokenizer, lookback_time=lookback_time, lookahead_time=lookahead_time),
    ).cpu()


class InferenceServer:
    def __init__(
            self,
            model,
            tokenizer,
            max_batch_size=8,
            batch_timeout=0.1,
            idle_timeout=2,
            socket_path=SOCKET_PATH
    ):
        """
        Initializes the inference server.
        :param model: The model to use for inference.
        :param tokenizer: The tokenizer to use for processing inputs.
        :param max_batch_size: Maximum batch size for processing requests.
        :param batch_timeout: Time in seconds to wait for more requests before processing a batch.
        :param idle_timeout: Time in seconds to wait before shutting down due to no clients.
        :param socket_path: The address used for IPC.
        """
        self.model: Mapperatorinator = model
        self.tokenizer: Tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.idle_timeout = idle_timeout
        self.socket_path = socket_path
        self.grouped_requests = {}  # holds pending requests
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.listener = None
        self.connections = 0

    def start(self):
        # Remove stale socket
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        # Start IPC listener
        self.listener = Listener(self.socket_path)
        threading.Thread(target=self._listener_thread, daemon=True).start()
        # Start batcher thread
        threading.Thread(target=self._batch_thread, daemon=True).start()
        # Start idle monitor
        threading.Thread(target=self._idle_monitor, daemon=True).start()

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
        with conn:
            while True:
                try:
                    model_kwargs, generate_kwargs = conn.recv()
                except EOFError:
                    break

                generate_kwargs_set = frozenset(generate_kwargs.items())

                # Prepare a response event
                response_event = threading.Event()
                batch_size = model_kwargs['inputs'].shape[0]
                record = {'model_kwargs': model_kwargs, 'total_work': batch_size, 'work_done': 0, 'conn': conn, 'event': response_event, 'result': None}

                # Enqueue request
                with self.lock:
                    if generate_kwargs_set in self.grouped_requests:
                        self.grouped_requests[generate_kwargs_set].append(record)
                    else:
                        self.grouped_requests[generate_kwargs_set] = [record]

                # Wait until batch thread processes it
                response_event.wait()

                # Send back result
                conn.send(record['result'])
        with self.lock:
            self.connections -= 1

    def _batch_thread(self):
        while not self.shutdown_flag.is_set():
            time.sleep(self.batch_timeout)
            with self.lock:
                if not self.grouped_requests:
                    continue
                generate_kwargs_set: frozenset = list(self.grouped_requests.keys())[0]
                requests: list = self.grouped_requests[generate_kwargs_set]

                generate_kwargs: dict = dict(generate_kwargs_set)
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

            outputs = model_generate(self.model, self.tokenizer, model_kwargs, generate_kwargs)

            # Split and dispatch results
            batch_i = 0
            for i, (_, request, work_done) in enumerate(batch_requests):
                padding = paddings[i]
                out = outputs[batch_i:batch_i + work_done, padding:]  # Remove padding from the left
                batch_i += work_done
                request['result'] = out if request['result'] is None else torch.cat((request['result'], out), dim=0)
                request['work_done'] += work_done
                if request['work_done'] >= request['total_work']:
                    # All work done for this record, signal completion
                    request['event'].set()

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
            batch_timeout=0.1,
            idle_timeout=2,
            socket_path=SOCKET_PATH,
    ):
        """
        Initializes the inference client. Automatically starts the inference server if it is not running.
        :param model_loader: Function to load the model.
        :param tokenizer_loader: Function to load the tokenizer.
        :param max_batch_size: Maximum batch size for processing requests.
        :param batch_timeout: Time in seconds to wait for more requests before processing a batch.
        :param idle_timeout: Time in seconds to wait before shutting down due to no clients.
        :param socket_path: The address used for IPC.
        """
        self.model_loader = model_loader
        self.tokenizer_loader = tokenizer_loader
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.idle_timeout = idle_timeout
        self.socket_path = socket_path
        self.conn = None

    def __enter__(self):
        try:
            self.conn = Client(self.socket_path)
        except FileNotFoundError:
            # No server: start one
            threading.Thread(target=self._start_server, args=(self.model_loader, self.tokenizer_loader), daemon=False).start()
            # Wait for server socket to appear
            while not os.path.exists(self.socket_path):
                time.sleep(0.1)
            self.conn = Client(self.socket_path)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.conn:
            self.conn.close()

    def _start_server(self, model_loader, tokenizer_loader):
        # Load model inside server process
        model = model_loader()
        tokenizer = tokenizer_loader()
        print(f"Model loaded: {model.name_or_path} on device {model.device}")
        server = InferenceServer(
            model,
            tokenizer,
            max_batch_size=self.max_batch_size,
            batch_timeout=self.batch_timeout,
            idle_timeout=self.idle_timeout,
            socket_path=self.socket_path
        )
        server.start()
        # Block until shutdown
        while not server.shutdown_flag.is_set():
            time.sleep(1)

    def generate(self, model_kwargs, generate_kwargs):
        # Send request and wait for response
        self.conn.send((model_kwargs, generate_kwargs))
        return self.conn.recv()


if __name__ == "__main__":
    ckpt_path_str = "OliBomby/Mapperatorinator-v30"

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
