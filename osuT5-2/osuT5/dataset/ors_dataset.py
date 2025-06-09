from __future__ import annotations

import json
import os
import random
from multiprocessing.managers import Namespace
from typing import Optional, Callable, TextIO
from pathlib import Path
import zipfile
import re
import math
import rosu_pp_py as rosu
import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig
from slider import Beatmap
from torch.utils.data import IterableDataset
from typing import BinaryIO
from pydub import AudioSegment
from .data_utils import  remove_events_of_type
from .osu_parser import OsuParser
from ..tokenizer import Event, EventType, Tokenizer, ContextType
import gc
import io

OSZ_FILE_EXTENSION = ".osz"
AUDIO_FILE_NAME = "audio.mp3"
MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100



class OrsDataset(IterableDataset):
    __slots__ = (
        "path",
        "start",
        "end",
        "args",
        "parser",
        "tokenizer",
        "beatmap_files",
        "test",
        "shared",
        "sample_weights",
    )

    def __init__(
            self,
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            beatmap_files: Optional[list[Path]] = None,
            test: bool = False,
            shared: Namespace = None,
    ):
        """Manage and process ORS dataset.

        Attributes:
            args: Data loading arguments.
            parser: Instance of OsuParser class.
            tokenizer: Instance of Tokenizer class.
            beatmap_files: List of beatmap files to process. Overrides track index range.
            test: Whether to load the test dataset.
        """
        super().__init__()
        self.path = args.test_dataset_path if test else args.train_dataset_path
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.beatmap_files = beatmap_files
        self.test = test
        self.shared = shared
        self.sample_weights = self._get_sample_weights(args.sample_weights_path)
    @staticmethod
    def _get_sample_weights(sample_weights_path):
        if not os.path.exists(sample_weights_path):
            return None

        # Load the sample weights csv to a dictionary
        with open(sample_weights_path, "r") as f:
            sample_weights = {int(line.split(",")[0]): np.clip(float(line.split(",")[1]), 0.1, 10) for line in f.readlines()}
            # Normalize the weights so the mean is 1
            mean = sum(sample_weights.values()) / len(sample_weights)
            sample_weights = {k: v / mean for k, v in sample_weights.items()}

        return sample_weights
    




            
    def _get_beatmap_files(self, seed=42) -> list[Path]:
        if self.beatmap_files is not None:
            return self.beatmap_files

        beatmap_files = []
        for track in Path(self.path).iterdir():
            if track.is_dir():
                track_idx = int(track.stem[5:])
                if self.start <= track_idx <= self.end:
                    beatmap_files.append(track)
        
        # Set random seed for reproducibility
        rng = np.random.RandomState(seed)
        beatmap_files = np.array(beatmap_files)
        rng.shuffle(beatmap_files)
        beatmap_files = beatmap_files.tolist()
                        
            
            
        worker_info = torch.utils.data.get_worker_info()  # Get worker-specific info
        if worker_info is not None:  # If there are actually multiple workers
            # Split the dataset among workers based on the number of workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(np.ceil(len(beatmap_files) / float(num_workers)))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(beatmap_files))
            beatmap_files = beatmap_files[start_idx:end_idx]
            print(f"Worker {worker_id}: {len(beatmap_files)} files assigned.")
 
        return beatmap_files


    def __iter__(self):

        beatmap_files = self._get_beatmap_files()
        """   
        if not self.test:
            # shuffle the beatmaps
            random.shuffle(beatmap_files)
            """
        if self.args.cycle_length > 1 and not self.test:
            return InterleavingBeatmapDatasetIterable(
                beatmap_files,
                self._iterable_factory,
                self.args.cycle_length,
            )

        return self._iterable_factory(beatmap_files).__iter__()


    def _iterable_factory(self, beatmap_files: list[Path]):
        return BeatmapDatasetIterable(
            beatmap_files,
            self.args,
            self.parser,
            self.tokenizer,
            self.test,
            self.shared,
            self.sample_weights,
        )


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
            self,
            beatmap_files: list[Path],
            iterable_factory: Callable,
            cycle_length: int,
    ):
        per_worker = int(np.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                beatmap_files[
                i * per_worker: min(len(beatmap_files), (i + 1) * per_worker)
                ]
            ).__iter__()
            for i in range(cycle_length)
        ]
        for i in range(cycle_length):
            worker_dataset = beatmap_files[i * per_worker: min(len(beatmap_files), (i + 1) * per_worker)]
            print(f"Worker {i}: {len(worker_dataset)} files assigned.")
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                self.workers.remove(self.workers[self.index])
            except Exception as e:
                print(f"Error in interleaving dataset iteration: {e}")
                continue
            


class BeatmapDatasetIterable:
    __slots__ = (
        "beatmap_files",
        "args",
        "parser",
        "tokenizer",
        "test",
        "shared",
        "frame_seq_len",
        "min_pre_token_len",
        "pre_token_len",
        "class_dropout_prob",
        "diff_dropout_prob",
        "add_pre_tokens",
        "add_empty_sequences",
        "log_dir",
        "sample_weights",
        "gen_start_frame",
        "gen_end_frame",
    )

    def __init__(
            self,
            beatmap_files: list[Path],
            args: DictConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            test: bool,
            shared: Namespace,
            sample_weights: dict[int, float] = None,
    ):
        self.beatmap_files = beatmap_files
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        self.sample_weights = sample_weights
        # let N = |src_seq_len|
        # N-1 frames creates N mel-spectrogram frames
        self.frame_seq_len = (args.src_seq_len) - 1
        self.gen_start_frame = int(round(args.lookback * self.frame_seq_len))
        self.gen_end_frame = int(round((1 - args.lookahead) * self.frame_seq_len))
        # let N = |tgt_seq_len|
        # [SOS] token + event_tokens + [EOS] token creates N+1 tokens
        # [SOS] token + event_tokens[:-1] creates N target sequence
        # event_tokens[1:] + [EOS] token creates N label sequence
        self.min_pre_token_len = 4
        self.pre_token_len = args.tgt_seq_len // 2
        self.class_dropout_prob = 1 if self.test else args.class_dropout_prob
        self.diff_dropout_prob = 0 if self.test else args.diff_dropout_prob
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences
        self.log_dir = args.log_dir
        
    def get_speed(self, max_speed: float = 1.5, min_speed: float = 0.9) -> float:
        if random.random() < 0.7: 
            return 1.0
        
        base = random.random()
        biased = math.pow(base, 0.5) 
        return min_speed + (max_speed - min_speed) * biased
    
    def load_audio_file(self, file, sample_rate: int, speed: float = 1.0) -> npt.NDArray:
        try:
            audio = AudioSegment.from_file(file, format=file.name.split(".")[-1])
            audio.frame_rate = int(audio.frame_rate * speed)
            audio = audio.set_frame_rate(sample_rate)
            audio = audio.set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
               
            del audio 
            gc.collect()
            
            return samples
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
        
    def parse_map_file(self, bmlines):
        LIST_SECTIONS = ["Events", "TimingPoints", "HitObjects"]
        cfg = {}
        section = None
        for l in bmlines:
            # comments
            if l.startswith("//"):
                continue

            # section end check
            if l.strip() == "":
                section = None
                continue

            # header check
            m = re.search(r"^\[(.*)\]$", l)
            if m is not None:
                section = m.group(1)
                if section in LIST_SECTIONS:
                    cfg[section] = []
                else:
                    cfg[section] = {}
                continue

            if section is None:
                continue

            if section in LIST_SECTIONS:
                cfg[section].append(l.strip())
            else:
                # key-value check
                m = re.search(r"^(\w*)\s?:\s?(.*)$", l)
                if m is not None:
                    cfg[section][m.group(1)] = m.group(2).strip()

        return cfg
    
    def _get_frames(self, samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Segment audio samples into frames.

        Each frame has `frame_size` audio samples.
        It will also calculate and return the time of each audio frame, in miliseconds.

        Args:
            samples: Audio time-series.

        Returns:
            frames: Audio frames.
            frame_times: Audio frame times.
        """
        samples = np.pad(samples, [0, self.args.hop_length - len(samples) % self.args.hop_length])
        frames = np.reshape(samples, (-1, self.args.hop_length))
        frames_per_milisecond = (
                self.args.sample_rate / self.args.hop_length / MILISECONDS_PER_SECOND
        )
        frame_times = np.arange(len(frames)) / frames_per_milisecond
        return frames, frame_times

    def _create_sequences(
            self,
            frames: npt.NDArray,
            frame_times: npt.NDArray,
            events: list[Event],
            other_events: Optional[list[Event]] = None,
            extra_data: Optional[dict] = None,
    ) -> list[dict[str, int | npt.NDArray | list[Event]]]:
        """Create frame and token sequences for training/testing.

        Args:
            events: Events and time shifts.
            frames: Audio frames.

        Returns:
            A list of source and target sequences.
        """

        def get_event_indices(events2: list[Event]) -> tuple[list[int], list[int]]:
            # Corresponding start event index for every audio frame.
            start_indices = []
            event_index = 0
            event_time = -np.inf

            for current_time in frame_times:
                while event_time < current_time and event_index < len(events2):
                    if events2[event_index].type == EventType.TIME_SHIFT:
                        event_time = events2[event_index].value
                    event_index += 1
                start_indices.append(event_index - 1)

            # Corresponding end event index for every audio frame.
            end_indices = start_indices[1:] + [len(events2)]

            return start_indices, end_indices

        event_start_indices, event_end_indices = get_event_indices(events)

        other_event_start_indices, other_event_end_indices = None, None
        if other_events is not None:
            other_event_start_indices, other_event_end_indices = get_event_indices(other_events)

        frame_seq_len = (self.args.src_seq_len) - 1

        gen_start_frame_x = int(round(0.2 * frame_seq_len)) if random.random() < 0.5 else 0
        gen_end_frame_x = frame_seq_len

        offset = random.randint(0, frame_seq_len) if random.random() < 0.5 else 0

        sequences = []
        n_frames = len(frames)

        # Divide audio frames into splits
        for frame_start_idx in range(offset, n_frames - gen_start_frame_x, frame_seq_len):
            frame_end_idx = min(frame_start_idx + frame_seq_len, n_frames)

            gen_start_frame = min(frame_start_idx + gen_start_frame_x, n_frames - 1)
            gen_end_frame = min(frame_start_idx + gen_end_frame_x, n_frames)

            event_start_idx = event_start_indices[frame_start_idx]
            gen_start_idx = event_start_indices[gen_start_frame]
            gen_end_idx = event_end_indices[gen_end_frame - 1]

            frame_pre_idx = max(frame_start_idx - frame_seq_len, 0)
            target_pre_idx = event_start_indices[frame_pre_idx]
            
            # Create the sequence
            sequence = {
                "time": frame_times[frame_start_idx],
                "frames": frames[frame_start_idx:frame_end_idx],
                'global_pos': torch.tensor([frame_start_idx / n_frames, frame_end_idx / n_frames], dtype=torch.float32),
                "events": events[event_start_idx:gen_end_idx],
                "labels_offset": gen_start_idx - event_start_idx,
            }
             
            # Add extra data to the sequence
            if extra_data is not None:
                sequence |= extra_data

            if self.args.add_pre_tokens or self.args.add_pre_tokens_at_step >= 0:
                sequence["pre_events"] = events[target_pre_idx:event_start_idx]

            if other_events is not None:
                other_target_start_idx = other_event_start_indices[frame_start_idx]
                other_target_end_idx = other_event_end_indices[frame_end_idx - 1]
                sequence["other_events"] = other_events[other_target_start_idx:other_target_end_idx]
            

            sequences.append(sequence)

        

        return sequences

    def _trim_time_shifts(self, sequence: dict) -> dict:
        """Make all time shifts in the sequence relative to the start time of the sequence,
        and normalize time values,
        and remove any time shifts for anchor events.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with trimmed time shifts.
        """

        def process(events: list[Event], start_time, offset=-1) -> list[Event] | tuple[list[Event], int]:
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    # We cant modify the event objects themselves because that will affect subsequent sequences
                    events[i] = Event(EventType.TIME_SHIFT, int((event.value - start_time) * STEPS_PER_MILLISECOND))

            # Loop through the events in reverse to remove any time shifts that occur before anchor events
            delete_next_time_shift = False
            for i in range(len(events) - 1, -1, -1):
                if events[i].type == EventType.TIME_SHIFT and delete_next_time_shift:
                    delete_next_time_shift = False
                    del events[i]
                    if i < offset:
                        offset -= 1
                    continue
                elif events[i].type in [EventType.BEZIER_ANCHOR, EventType.PERFECT_ANCHOR, EventType.CATMULL_ANCHOR,
                                        EventType.RED_ANCHOR]:
                    delete_next_time_shift = True

            if offset >= 0:
                return events, offset

            return events

        start_time = sequence["time"]
        del sequence["time"]

        sequence["events"], sequence["labels_offset"] = process(sequence["events"], start_time, sequence["labels_offset"])

        if "pre_events" in sequence:
            sequence["pre_events"] = process(sequence["pre_events"], start_time)

        if "other_events" in sequence:
            sequence["other_events"] = process(sequence["other_events"], start_time)

        return sequence

    def _tokenize_sequence(self, sequence: dict) -> dict:
        """Tokenize the event sequence.

        Begin token sequence with `[SOS]` token (start-of-sequence).
        End token sequence with `[EOS]` token (end-of-sequence).

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with tokenized events.
        """
        tokens = torch.empty(len(sequence["events"]) + 2, dtype=torch.long)
        tokens[0] = self.tokenizer.sos_id
        for i, event in enumerate(sequence["events"]):
            tokens[i + 1] = self.tokenizer.encode(event)
        tokens[-1] = self.tokenizer.eos_id
        sequence["tokens"] = tokens
        del sequence["events"]
            
        sequence["difficulty"] = torch.tensor(sequence["difficulty"], dtype=torch.float32)
         
        if self.test:
            sequence["mapper_idx"] = self.tokenizer.encode_mapper_idx(sequence["beatmap_id"])
        elif random.random() >= self.args.mapper_idx_dropout_prob:
            sequence["mapper_idx"] = self.tokenizer.encode_mapper_idx(sequence["beatmap_id"])
        else:
            sequence["mapper_idx"] = -1

        del sequence["beatmap_id"]

        return sequence

    def _pad_and_split_token_sequence(self, sequence: dict) -> dict:
        """Pad token sequence to a fixed length and split decoder input and labels.

        Pad with `[PAD]` tokens until `tgt_seq_len`.

        Token sequence (w/o last token) is the input to the transformer decoder,
        token sequence (w/o first token) is the label, a.k.a. decoder ground truth.

        Prefix the token sequence with the pre_tokens sequence.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded tokens.
        """
        stl = self.args.special_token_len

        if "descriptor_tokens" in sequence:
            stl += len(sequence["descriptor_tokens"])

        tokens = sequence["tokens"]
        labels_offset = sequence["labels_offset"]
        pre_tokens = sequence["pre_tokens"] if "pre_tokens" in sequence else torch.empty(0, dtype=tokens.dtype)
        num_pre_tokens = len(pre_tokens) if self.args.add_pre_tokens else 0

        if self.args.max_pre_token_len > 0:
            num_pre_tokens = min(num_pre_tokens, self.args.max_pre_token_len)

        other_tokens = sequence["other_tokens"] if "other_tokens" in sequence else torch.empty(0, dtype=tokens.dtype)
        num_other_tokens = len(other_tokens) if "other_tokens" in sequence else 0

        ctl = 0 if sequence["context_type"] is None else 2
        if "other_tokens" in sequence:
            if "other_beatmap_id" in sequence:
                ctl += self.args.special_token_len

                if "other_descriptor_tokens" in sequence:
                    ctl += len(sequence["other_descriptor_tokens"])

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=tokens.dtype,
                                  device=tokens.device)
        label_tokens = torch.full((self.args.tgt_seq_len,), LABEL_IGNORE_ID, dtype=tokens.dtype, device=tokens.device)

        if self.args.center_pad_decoder:
            n = min(self.args.tgt_seq_len - self.pre_token_len, len(tokens) - 1)
            m = min(self.pre_token_len - stl - ctl, num_pre_tokens)
            o = min(self.pre_token_len - m - stl - ctl, num_other_tokens)
            si = self.pre_token_len - m - stl - ctl - o
        else:
            # n + m + special_token_length + num_other_tokens + padding = tgt_seq_len
            n = min(self.args.tgt_seq_len - stl - ctl - min(self.min_pre_token_len, num_pre_tokens),
                    len(tokens) - 1)
            m = min(self.args.tgt_seq_len - n - stl - ctl, num_pre_tokens)
            o = min(self.args.tgt_seq_len - n - stl - ctl - m, num_other_tokens)
            si = 0

        if sequence["context_type"] is not None:
            input_tokens[si] = self.tokenizer.context_sos[sequence["context_type"]]
            si += 1

        if "other_tokens" in sequence:
            if "other_beatmap_id" in sequence:
                if "other_beatmap_idx_token" in sequence:
                    input_tokens[si + self.args.style_token_index] = sequence["other_beatmap_idx_token"]
                    del sequence["other_beatmap_idx_token"]
                if "other_difficulty_token" in sequence:
                    input_tokens[si + self.args.diff_token_index] = sequence["other_difficulty_token"]
                    del sequence["other_difficulty_token"]
                if "other_mapper_token" in sequence:
                    input_tokens[si + self.args.mapper_token_index] = sequence["other_mapper_token"]
                    del sequence["other_mapper_token"]

                si += self.args.special_token_len

                if "other_descriptor_tokens" in sequence:
                    for token in sequence["other_descriptor_tokens"]:
                        input_tokens[si] = token
                        si += 1
                    del sequence["other_descriptor_tokens"]

            input_tokens[si:si + o] = other_tokens[:o]
            si += o

        if sequence["context_type"] is not None:
            input_tokens[si] = self.tokenizer.context_eos[sequence["context_type"]]
            si += 1

        if "beatmap_idx_token" in sequence:
            input_tokens[si + self.args.style_token_index] = sequence["beatmap_idx_token"]
            del sequence["beatmap_idx_token"]
        if "difficulty_token" in sequence:
            input_tokens[si + self.args.diff_token_index] = sequence["difficulty_token"]
            del sequence["difficulty_token"]
        if "mapper_token" in sequence:
            input_tokens[si + self.args.mapper_token_index] = sequence["mapper_token"]
            del sequence["mapper_token"]

        si += self.args.special_token_len

        if "descriptor_tokens" in sequence:
            for token in sequence["descriptor_tokens"]:
                input_tokens[si] = token
                si += 1
            del sequence["descriptor_tokens"]

        if m > 0:
            input_tokens[si:si + m] = pre_tokens[-m:]

        input_tokens[si + m:si + m + n] = tokens[:n]
        label_tokens[si + m + labels_offset:si + m + n] = tokens[1 + labels_offset:1 + n]

        # Randomize some input tokens
        def randomize_tokens(tokens):
            offset = torch.randint(low=-self.args.timing_random_offset, high=self.args.timing_random_offset+1, size=tokens.shape)
            return torch.where((self.tokenizer.event_start[EventType.TIME_SHIFT] <= tokens) & (
                    tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
                                       torch.clamp(tokens + offset,
                                                   self.tokenizer.event_start[EventType.TIME_SHIFT],
                                                   self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
                                       tokens)

        if self.args.timing_random_offset > 0:
            input_tokens[si:si + m + n] = randomize_tokens(input_tokens[si:si + m + n])
        # input_tokens = torch.where((self.tokenizer.event_start[EventType.DISTANCE] <= input_tokens) & (input_tokens < self.tokenizer.event_end[EventType.DISTANCE]),
        #                               torch.clamp(input_tokens + torch.randint_like(input_tokens, -10, 10), self.tokenizer.event_start[EventType.DISTANCE], self.tokenizer.event_end[EventType.DISTANCE] - 1),
        #                               input_tokens)

        sequence["decoder_input_ids"] = input_tokens
        sequence["decoder_attention_mask"] = input_tokens != self.tokenizer.pad_id
        sequence["labels"] = label_tokens

        del sequence["context_type"]
        del sequence["tokens"]
        del sequence["labels_offset"]
        if "pre_tokens" in sequence:
            del sequence["pre_tokens"]

        if "other_tokens" in sequence:
            del sequence["other_tokens"]

        if "other_beatmap_id" in sequence:
            del sequence["other_beatmap_id"]

        return sequence


    def _pad_frame_sequence(self, sequence: dict) -> dict:
        """Pad frame sequence with zeros until `frame_seq_len`.

        Frame sequence can be further processed into Mel spectrogram frames,
        which is the input to the transformer encoder.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with padded frames.
        """
        frames = torch.from_numpy(sequence["frames"]).to(torch.float32)
        main_seq_len = self.args.src_seq_len - 1
        if frames.shape[0] != main_seq_len:
            n = min(main_seq_len, len(frames))
            padded_frames = torch.zeros(
                main_seq_len,
                frames.shape[-1],
                dtype=frames.dtype,
                device=frames.device,
            )
            padded_frames[:n] = frames[:n]
            sequence["frames"] = torch.flatten(padded_frames)
        else:
            sequence["frames"] = torch.flatten(frames)



        return sequence

    def maybe_change_dataset(self):
        if self.shared is None:
            return
        step = self.shared.current_train_step
        if 0 <= self.args.add_empty_sequences_at_step <= step and not self.add_empty_sequences:
            self.add_empty_sequences = True
        if 0 <= self.args.add_pre_tokens_at_step <= step and not self.add_pre_tokens:
            self.add_pre_tokens = True

    def __iter__(self):
        return self._get_next_beatmaps()

    @staticmethod
    def _load_metadata(track_path: Path) -> dict:
        metadata_file = track_path / "metadata.json"
        with open(metadata_file) as f:
            return json.load(f)

    @staticmethod
    def _get_difficulty(metadata: dict, beatmap_name: str, double_time: bool = False):
        if double_time:
            return metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["64"]
        return metadata["Beatmaps"][beatmap_name]["StandardStarRating"]["0"]

    @staticmethod
    def _get_idx(metadata: dict, beatmap_name: str):
        return metadata["Beatmaps"][beatmap_name]["Index"]


            
    def _get_next_beatmaps(self) -> dict:

        def get_difficulty(content, speed):
            map = rosu.Beatmap(content=content)
            perf = rosu.Performance(clock_rate=speed)
            attrs = perf.calculate(map)
            return attrs.difficulty.stars

        for track_path in self.beatmap_files:
            
            speed = self.get_speed(max_speed=1.2, min_speed=1.0) if not self.test else 1.0

            audio_filename = track_path / "audio.mp3"
            if not audio_filename.exists():
                audio_filename = track_path / "audio.ogg"
                if not audio_filename.exists():
                    print(f"Audio file not found for {track_path}")
                    continue

            with audio_filename.open("rb") as f2:
                audio_samples = self.load_audio_file(f2, self.args.sample_rate, speed)
                f2.close()

            if audio_samples is None:
                print(f"Audio samples returned None for {track_path}")
                continue

            for file in (track_path / "beatmaps").iterdir():
                with io.TextIOWrapper(file.open("rb"), encoding="utf-8-sig") as f:
                    content = f.read()
                    beatmap = Beatmap.parse(content)
                    cfg = self.parse_map_file(content.splitlines())
                    difficulty = get_difficulty(content, speed)
                    beatmap_id = int(cfg["Metadata"]["BeatmapID"])
                    beatmapset_id = cfg["Metadata"]["BeatmapSetID"]

                    if difficulty < 1 or difficulty > 10:
                        continue

                    for sample in self._get_next_beatmap(audio_samples, beatmap, difficulty, beatmap_id, beatmapset_id, speed):
                        yield sample


    def _get_next_beatmap(self, audio_samples, beatmap, difficulty: float, beatmap_id: int, beatmapset_id: int, speed: float) -> dict:
        context_type = None
        
        frames, frame_times = self._get_frames(audio_samples)
        
        flip, mirror = False, False
        """
        if random.random() < 0.3 and not self.test:
            # 50% chance of flipping and 50% chance of mirroring
            flip = random.random() < 0.5
            mirror = not flip
        """
        events = self.parser.parse(beatmap, speed, flip=flip, mirror=mirror)
        extra_data = {
            "context_type": context_type,
            "beatmap_id": beatmap_id,
            "beatmap_idx": -1,
            "difficulty": difficulty,
        }

        other_events = None

        sequences = self._create_sequences(
            frames,
            frame_times,
            events,
            other_events,
            extra_data,
        )
        for i, sequence in enumerate(sequences):
          
            self.maybe_change_dataset()
            sequence = self._trim_time_shifts(sequence)
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            if not self.add_empty_sequences and ((sequence["labels"] == self.tokenizer.eos_id) | (
                    sequence["labels"] == LABEL_IGNORE_ID)).all():
                continue
            del sequence["beatmap_idx"]
            yield sequence
                

