from __future__ import annotations

import os
import random
from datetime import datetime
from multiprocessing.managers import Namespace
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import Series, DataFrame
from slider import Beatmap
from torch.utils.data import IterableDataset

from .data_utils import load_audio_file, remove_events_of_type, get_hold_note_ratio, get_scroll_speed_ratio, \
    get_hitsounded_status, get_song_length
from .osu_parser import OsuParser
from ..tokenizer import Event, EventType, Tokenizer, ContextType
from ..config import DataConfig

OSZ_FILE_EXTENSION = ".osz"
AUDIO_FILE_NAME = "audio.mp3"
MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100
context_types_with_kiai = [ContextType.NO_HS, ContextType.GD, ContextType.MAP]


class MmrsDataset(IterableDataset):
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
            args: DataConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            subset_ids: Optional[list[int]] = None,
            test: bool = False,
            shared: Namespace = None,
    ):
        """Manage and process MMRS dataset.

        Attributes:
            args: Data loading arguments.
            parser: Instance of OsuParser class.
            tokenizer: Instance of Tokenizer class.
            subset_ids: List of beatmap set IDs to process. Overrides track index range.
            test: Whether to load the test dataset.
        """
        super().__init__()
        self._validate_args(args)
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        self.path = Path(args.test_dataset_path if test else args.train_dataset_path)
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.metadata = self._load_metadata()
        if subset_ids is not None:
            self.subset_ids = subset_ids
            self.start = 0
            self.end = len(subset_ids)
        else:
            self.subset_ids = self._beatmap_set_ids_from_metadata()
        self.sample_weights = self._get_sample_weights(args.sample_weights_path)

    def _validate_args(self, args: DataConfig):
        if not args.per_track:
            raise ValueError("MMRS dataset requires per_track to be True")
        if args.only_last_beatmap:
            raise ValueError("MMRS dataset does not support only_last_beatmap")

    def _load_metadata(self):
        # Loads the metadata parquet from the dataset path
        df = pd.read_parquet(self.path / "metadata.parquet")
        df["BeatmapIdx"] = df.index
        df.set_index(["BeatmapSetId", "Id"], inplace=True)
        df.sort_index(inplace=True)
        df = df[df["ModeInt"].isin(self.args.gamemodes)]
        if self.args.min_year:
            df = df[df["RankedDate"] >= datetime(self.args.min_year, 1, 1)]
        return df

    def _beatmap_set_ids_from_metadata(self):
        return self.metadata.index.to_frame()["BeatmapSetId"].unique().tolist()

    @staticmethod
    def _get_sample_weights(sample_weights_path):
        if not os.path.exists(sample_weights_path):
            return None

        # Load the sample weights csv to a dictionary
        with open(sample_weights_path, "r") as f:
            sample_weights = {int(line.split(",")[0]): np.clip(float(line.split(",")[1]), 0.1, 10) for line in
                              f.readlines()}
            # Normalize the weights so the mean is 1
            mean = sum(sample_weights.values()) / len(sample_weights)
            sample_weights = {k: v / mean for k, v in sample_weights.items()}

        return sample_weights

    def __iter__(self):
        subset_ids = self.subset_ids[self.start:self.end].copy()

        if not self.test:
            random.shuffle(subset_ids)

        if self.args.cycle_length > 1 and not self.test:
            return InterleavingBeatmapDatasetIterable(
                subset_ids,
                self._iterable_factory,
                self.args.cycle_length,
            )

        return self._iterable_factory(subset_ids).__iter__()

    def _iterable_factory(self, subset_ids: list[int]):
        return BeatmapDatasetIterable(
            subset_ids,
            self.args,
            self.path,
            self.metadata,
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
            subset_ids: list[int],
            iterable_factory: Callable,
            cycle_length: int,
    ):
        per_worker = int(np.ceil(len(subset_ids) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                subset_ids[i * per_worker: min(len(subset_ids), (i + 1) * per_worker)]
            ).__iter__()
            for i in range(cycle_length)
        ]
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
        raise StopIteration


class BeatmapDatasetIterable:
    __slots__ = (
        "subset_ids",
        "args",
        "path",
        "metadata",
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
        "sample_weights",
        "gen_start_frame",
        "gen_end_frame",
        "lookback_allowed",
    )

    def __init__(
            self,
            subset_ids: list[int],
            args: DataConfig,
            path: Path,
            metadata: pd.DataFrame,
            parser: OsuParser,
            tokenizer: Tokenizer,
            test: bool,
            shared: Namespace,
            sample_weights: dict[int, float] = None,
    ):
        self.subset_ids = subset_ids
        self.args = args
        self.path = path
        self.metadata = metadata
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        self.sample_weights = sample_weights
        # let N = |src_seq_len|
        # N-1 frames creates N mel-spectrogram frames
        self.frame_seq_len = args.src_seq_len - 1
        self.lookback_allowed = all(len(c["out"]) == 1 for c in args.context_types)
        if not self.lookback_allowed and args.lookback > 0 and args.lookback_prob > 0:
            raise ValueError("Lookback is currently not supported for multiple output contexts.")
        # let N = |tgt_seq_len|
        # [SOS] token + event_tokens + [EOS] token creates N+1 tokens
        # [SOS] token + event_tokens[:-1] creates N target sequence
        # event_tokens[1:] + [EOS] token creates N label sequence
        self.min_pre_token_len = 4
        self.pre_token_len = args.tgt_seq_len // 2
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences

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
            out_context: list[dict],
            in_context: list[dict],
            extra_data: Optional[dict] = None,
    ) -> list[dict[str, int | npt.NDArray | list[Event] | list[dict]]]:
        """Create frame and token sequences for training/testing.

        Args:
            frames: Audio frames.

        Returns:
            A list of source and target sequences.
        """

        def get_event_indices(events2: list[Event], event_times2: list[int]) -> tuple[list[int], list[int]]:
            if len(events2) == 0:
                return [], []

            # Corresponding start event index for every audio frame.
            start_indices = []
            event_index = 0

            for current_time in frame_times:
                while event_index < len(events2) and event_times2[event_index] < current_time:
                    event_index += 1
                start_indices.append(event_index)

            # Corresponding end event index for every audio frame.
            end_indices = start_indices[1:] + [start_indices[-1]]

            return start_indices, end_indices

        start_indices, end_indices = {}, {}
        for context in in_context + out_context:
            (start_indices[context["extra"]["id"]], end_indices[context["extra"]["id"]]) = (
                get_event_indices(context["events"], context["event_times"]))

        sequences = []
        n_frames = len(frames)
        offset = random.randint(0, min(self.frame_seq_len, 2000)) if not self.test and random.random() < self.args.frame_offset_augment_prob else 0
        gen_start_frame_x = int(round(self.args.lookback * self.frame_seq_len)) if not self.test and self.lookback_allowed and random.random() < self.args.lookback_prob else 0
        gen_end_frame_x = int(round((1 - self.args.lookahead) * self.frame_seq_len))
        last_kiai = {}
        last_sv = {}

        # Divide audio frames into splits
        for frame_start_idx in range(offset, n_frames - gen_start_frame_x, self.frame_seq_len):
            frame_end_idx = min(frame_start_idx + self.frame_seq_len, n_frames)

            gen_start_frame = min(frame_start_idx + gen_start_frame_x, n_frames - 1)
            gen_end_frame = min(frame_start_idx + gen_end_frame_x, n_frames)

            # Assumes only one output context since
            event_start_idx = start_indices[out_context[0]["extra"]["id"]][frame_start_idx]
            gen_start_idx = start_indices[out_context[0]["extra"]["id"]][gen_start_frame]

            frame_pre_idx = max(frame_start_idx - self.frame_seq_len, 0)

            def slice_events(context, frame_start_idx, frame_end_idx):
                if len(context["events"]) == 0:
                    return []
                identifier = context["extra"]["id"]
                event_start_idx = start_indices[identifier][frame_start_idx]
                event_end_idx = end_indices[identifier][frame_end_idx - 1]
                return context["events"][event_start_idx:event_end_idx]

            def slice_context(context, frame_start_idx, frame_end_idx):
                result = {"events": slice_events(context, frame_start_idx, frame_end_idx)} | context["extra"]
                result["time"] = frame_times[frame_start_idx]
                return result

            # Create the sequence
            sequence: dict[str, str | int | list[Event] | dict] = {
                           "frames": frames[frame_start_idx:frame_end_idx],
                           "labels_offset": gen_start_idx - event_start_idx,
                           "out_context": [slice_context(context, frame_start_idx, gen_end_frame) for context in
                                           out_context],
                           "in_context": [slice_context(context, frame_start_idx, frame_end_idx) for context in
                                          in_context],
                           "song_position": torch.tensor([frame_start_idx / n_frames, frame_end_idx / n_frames], dtype=torch.float32),
                       } | extra_data

            sequence["special"] = sequence["special"].copy()
            sequence["special"]["time"] = frame_times[frame_start_idx]

            if self.args.add_pre_tokens or self.args.add_pre_tokens_at_step >= 0:
                sequence["pre_events"] = slice_events(out_context[0], frame_pre_idx, frame_start_idx)

            def add_last_kiai(sequence_context, last_kiai):
                if (sequence_context["context_type"] != ContextType.KIAI and
                        not (self.args.add_kiai and sequence_context["context_type"] in [ContextType.GD, ContextType.MAP])):
                    return
                if sequence_context["id"] in last_kiai:
                    sequence_context["last_kiai"] = last_kiai[sequence_context["id"]]
                else:
                    sequence_context["last_kiai"] = Event(EventType.KIAI, 0)
                # Find the last kiai event in the out context
                for event in reversed(sequence_context["events"]):
                    if event.type == EventType.KIAI:
                        last_kiai[sequence_context["id"]] = event
                        break

            if self.args.add_kiai_special_token:
                for sequence_context in sequence["in_context"]:
                    add_last_kiai(sequence_context, last_kiai)
                for sequence_context in sequence["out_context"]:  # type: dict
                    add_last_kiai(sequence_context, last_kiai)
                    if "last_kiai" in sequence_context:
                        sequence["special"]["last_kiai"] = sequence_context["last_kiai"]

            def add_last_sv(sequence_context, last_sv):
                if (sequence_context["context_type"] != ContextType.SV and
                        not ((self.args.add_sv or self.args.add_mania_sv) and sequence_context["context_type"] in [ContextType.GD, ContextType.MAP])):
                    return
                if sequence_context["id"] in last_sv:
                    sequence_context["last_sv"] = last_sv[sequence_context["id"]]
                else:
                    sequence_context["last_sv"] = Event(EventType.SCROLL_SPEED, 100)
                # Find the last sv event in the out context
                for event in reversed(sequence_context["events"]):
                    if event.type == EventType.SCROLL_SPEED:
                        last_sv[sequence_context["id"]] = event
                        break

            if self.args.add_sv_special_token:
                for sequence_context in sequence["in_context"]:
                    add_last_sv(sequence_context, last_sv)
                for sequence_context in sequence["out_context"]:  # type: dict
                    add_last_sv(sequence_context, last_sv)
                    if "last_sv" in sequence_context:
                        sequence["special"]["last_sv"] = sequence_context["last_sv"]

            sequences.append(sequence)

        return sequences

    def _normalize_time_shifts(self, sequence: dict, beatmap_path) -> dict:
        """Make all time shifts in the sequence relative to the start time of the sequence,
        and normalize time values.

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with trimmed time shifts.
        """

        min_t = self.tokenizer.event_range[EventType.TIME_SHIFT].min_value
        max_t = self.tokenizer.event_range[EventType.TIME_SHIFT].max_value

        def process(events: list[Event], start_time) -> list[Event] | tuple[list[Event], int]:
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    # We cant modify the event objects themselves because that will affect subsequent sequences
                    t = int((event.value - start_time) * STEPS_PER_MILLISECOND)
                    if t < min_t or t > max_t:
                        print(f"WARNING: Time shift out of range ({t}) in beatmap {beatmap_path}")
                        t = np.clip(t, min_t, max_t)
                    events[i] = Event(EventType.TIME_SHIFT, t)

            return events

        if "pre_events" in sequence:
            sequence["pre_events"] = process(sequence["pre_events"], sequence["out_context"]["time"])

        for context in sequence["in_context"] + sequence["out_context"]:
            context["events"] = process(context["events"], context["time"])

        return sequence

    def _get_special_tokens(self, context: dict) -> list:
        special_tokens = []

        if "beatmap_id" in context:
            if self.args.add_gamemode_token:
                special_tokens.append(self.tokenizer.encode_gamemode(context["gamemode"]))

            if self.args.add_style_token:
                special_tokens.append(self.tokenizer.encode_style_idx(context["beatmap_idx"])
                                      if self.test or random.random() >= self.args.class_dropout_prob else self.tokenizer.style_unk)

            if self.args.add_diff_token:
                special_tokens.append(self.tokenizer.encode_diff(context["difficulty"])
                                      if self.test or random.random() >= self.args.diff_dropout_prob else self.tokenizer.diff_unk)

            if self.args.add_mapper_token:
                special_tokens.append(self.tokenizer.encode_mapper(context["beatmap_id"])
                                      if self.test or random.random() >= self.args.mapper_dropout_prob else self.tokenizer.mapper_unk)

            if self.args.add_year_token:
                special_tokens.append(self.tokenizer.encode_year(context["year"])
                                      if self.test or random.random() >= self.args.year_dropout_prob else self.tokenizer.year_unk)

            if self.args.add_hitsounded_token:
                special_tokens.append(self.tokenizer.encode(Event(EventType.HITSOUNDED, int(context["hitsounded"]))))

            if self.args.add_song_length_token:
                special_tokens.append(self.tokenizer.encode_song_length(context["song_length"]))

            if self.args.add_global_sv_token and "global_sv" in context:
                special_tokens.append(self.tokenizer.encode_global_sv(context["global_sv"]))

            if self.args.add_cs_token and "circle_size" in context:
                special_tokens.append(self.tokenizer.encode_cs(context["circle_size"])
                                      if self.test or random.random() >= self.args.cs_dropout_prob else self.tokenizer.cs_unk)

            if self.args.add_keycount_token and "keycount" in context:
                special_tokens.append(self.tokenizer.encode(Event(EventType.MANIA_KEYCOUNT, context["keycount"])))

            if self.args.add_hold_note_ratio_token and "hold_note_ratio" in context:
                special_tokens.append(self.tokenizer.encode_hold_note_ratio(context["hold_note_ratio"])
                                      if self.test or random.random() >= self.args.hold_note_ratio_dropout_prob else self.tokenizer.hold_note_ratio_unk)

            if self.args.add_scroll_speed_ratio_token and "scroll_speed_ratio" in context:
                special_tokens.append(self.tokenizer.encode_scroll_speed_ratio(context["scroll_speed_ratio"])
                                      if self.test or random.random() >= self.args.scroll_speed_ratio_dropout_prob else self.tokenizer.scroll_speed_ratio_unk)

            if self.args.add_descriptors:
                special_tokens.extend(self.tokenizer.encode_descriptor(context["beatmap_id"])
                                      if self.test or random.random() >= self.args.descriptor_dropout_prob else [
                    self.tokenizer.descriptor_unk])

            if self.args.add_kiai_special_token and "last_kiai" in context:
                special_tokens.append(self.tokenizer.encode(context["last_kiai"]))

            if self.args.add_sv_special_token and "last_sv" in context:
                special_tokens.append(self.tokenizer.encode(context["last_sv"]))

            if self.args.add_song_position_token:
                special_tokens.append(self.tokenizer.encode_song_position(context["time"], context["song_length"]))

        return special_tokens

    def _tokenize_sequence(self, sequence: dict) -> dict:
        """Tokenize the event sequence.

        Begin token sequence with `[SOS]` token (start-of-sequence).
        End token sequence with `[EOS]` token (end-of-sequence).

        Args:
            sequence: The input sequence.

        Returns:
            The same sequence with tokenized events.
        """
        sequence["special_tokens"] = self._get_special_tokens(sequence["special"])

        for context in sequence["in_context"] + sequence["out_context"]:
            tokens = torch.empty(len(context["events"]), dtype=torch.long)
            for i, event in enumerate(context["events"]):
                tokens[i] = self.tokenizer.encode(event)
            context["tokens"] = tokens
            context["special_tokens"] = self._get_special_tokens(context)

        if "pre_events" in sequence:
            pre_tokens = torch.empty(len(sequence["pre_events"]), dtype=torch.long)
            for i, event in enumerate(sequence["pre_events"]):
                pre_tokens[i] = self.tokenizer.encode(event)
            sequence["pre_tokens"] = pre_tokens
            del sequence["pre_events"]

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
        # Count irreducable tokens for SOS/EOS tokens
        stl = 1

        # Count irreducable tokens for all contexts
        stl += len(sequence["special_tokens"])
        for context in sequence["in_context"] + sequence["out_context"]:
            if context["add_type"]:
                stl += 2

            stl += len(context["special_tokens"])

        # Count reducible tokens, pre_tokens and context tokens
        num_tokens = sum(len(context["tokens"]) for context in sequence["out_context"])
        num_pre_tokens = len(sequence["pre_tokens"]) if "pre_tokens" in sequence else 0

        if self.args.max_pre_token_len > 0:
            num_pre_tokens = min(num_pre_tokens, self.args.max_pre_token_len)

        num_other_tokens = sum(len(context["tokens"]) for context in sequence["in_context"])

        # Trim tokens to target sequence length
        if self.args.center_pad_decoder:
            n = min(self.args.tgt_seq_len - self.pre_token_len - 1, num_tokens)
            m = min(self.pre_token_len - stl + 1, num_pre_tokens)
            o = min(self.pre_token_len - m - stl + 1, num_other_tokens)
            si = self.pre_token_len - m - stl + 1 - o
        else:
            # n + m + stl + o + padding = tgt_seq_len
            n = min(self.args.tgt_seq_len - stl - min(self.min_pre_token_len, num_pre_tokens), num_tokens)
            m = min(self.args.tgt_seq_len - stl - n, num_pre_tokens)
            o = min(self.args.tgt_seq_len - stl - n - m, num_other_tokens)
            si = 0

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=torch.long)
        label_tokens = torch.full((self.args.tgt_seq_len,), LABEL_IGNORE_ID, dtype=torch.long)

        def add_special_tokens(special_tokens, si):
            for token in special_tokens:
                input_tokens[si] = token
                si += 1
            return si

        def add_context(context, si, max_tokens):
            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_sos[context["context_type"]]
                si += 1

            si = add_special_tokens(context["special_tokens"], si)

            num_other_tokens_to_add = min(len(context["tokens"]), max_tokens)
            input_tokens[si:si + num_other_tokens_to_add] = context["tokens"][:num_other_tokens_to_add]
            si += num_other_tokens_to_add
            max_tokens -= num_other_tokens_to_add

            if context["add_type"]:
                input_tokens[si] = self.tokenizer.context_eos[context["context_type"]]
                si += 1

            return si, max_tokens

        for context in sequence["in_context"]:
            si, o = add_context(context, si, o)

        si = add_special_tokens(sequence["special_tokens"], si)
        start_random_index = si

        if m > 0:
            input_tokens[si:si + m] = sequence["pre_tokens"][-m:]
            si += m

        input_tokens[si] = self.tokenizer.sos_id
        si += 1
        start_label_index = si
        for context in sequence["out_context"]:
            si, n = add_context(context, si, n)
        end_index = si

        labels_offset = sequence["labels_offset"]
        label_tokens[start_label_index + labels_offset - 1:end_index - 1] = input_tokens[start_label_index + labels_offset:end_index]
        label_tokens[end_index - 1] = self.tokenizer.eos_id

        # Randomize some input tokens
        def randomize_tokens(tokens):
            offset_tokens = tokens.clone()

            if random.random() < self.args.timing_random_offset_prob:
                offset_tokens += torch.randint(low=-self.args.timing_random_offset, high=self.args.timing_random_offset + 1,
                                               size=tokens.shape)

            if random.random() < self.args.timing_random_offset_prob:
                offset_tokens += torch.randint(low=-self.args.timing_random_offset_2, high=self.args.timing_random_offset_2 + 1,
                                               size=(1,))

            return torch.where((self.tokenizer.event_start[EventType.TIME_SHIFT] <= tokens) & (
                    tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
                               torch.clamp(offset_tokens,
                                           self.tokenizer.event_start[EventType.TIME_SHIFT],
                                           self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
                               tokens)

        if self.args.timing_random_offset > 0 or self.args.timing_random_offset_2 > 0:
            input_tokens[start_random_index:end_index] = randomize_tokens(input_tokens[start_random_index:end_index])
        # input_tokens = torch.where((self.tokenizer.event_start[EventType.DISTANCE] <= input_tokens) & (input_tokens < self.tokenizer.event_end[EventType.DISTANCE]),
        #                               torch.clamp(input_tokens + torch.randint_like(input_tokens, -10, 10), self.tokenizer.event_start[EventType.DISTANCE], self.tokenizer.event_end[EventType.DISTANCE] - 1),
        #                               input_tokens)

        sequence["decoder_input_ids"] = input_tokens
        # sequence["decoder_attention_mask"] = input_tokens != self.tokenizer.pad_id
        sequence["labels"] = label_tokens

        del sequence["out_context"]
        del sequence["in_context"]
        del sequence["labels_offset"]
        del sequence["special_tokens"]
        del sequence["special"]
        if "pre_tokens" in sequence:
            del sequence["pre_tokens"]

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

        if frames.shape[0] != self.frame_seq_len:
            n = min(self.frame_seq_len, len(frames))
            padded_frames = torch.zeros(
                self.frame_seq_len,
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
        return self._get_next_tracks()

    def _get_difficulty(self, beatmap_metadata: Series, speed: float = 1.0) -> float:
        # StarRating is an array that gives the difficulty for the speeds:
        # 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        # Linearly interpolate between the two closest speeds
        star_ratings = beatmap_metadata["StarRating"]
        speed_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        return np.interp(speed, speed_ratios, star_ratings)  # type: ignore

    def _get_speed_augment(self):
        if self.test or random.random() >= self.args.dt_augment_prob:
            return 1.0

        mi, ma = self.args.dt_augment_range
        base = random.random()
        if self.args.dt_augment_sqrt:
            base = np.power(base, 0.5)
        return mi + (ma - mi) * base

    def _get_next_tracks(self) -> dict:
        for beatmapset_id in self.subset_ids:
            metadata = self.metadata.loc[beatmapset_id]

            if self.args.add_gd_context and len(metadata) <= 1:
                continue

            if all(beatmap_metadata["DifficultyRating"] < self.args.min_difficulty or
                   beatmap_metadata["DifficultyRating"] > self.args.max_difficulty for _, beatmap_metadata in metadata.iterrows()):
                continue

            speed = self._get_speed_augment()
            track_path = self.path / "data" / metadata.iloc[0]["BeatmapSetFolder"]
            audio_path = track_path / metadata.iloc[0]["AudioFile"]
            try:
                audio_samples = load_audio_file(audio_path, self.args.sample_rate, speed, self.args.normalize_audio)
            except Exception as e:
                print(f"Failed to load audio file: {audio_path}")
                print(e)
                continue

            for i, beatmap_metadata in metadata.iterrows():
                if (beatmap_metadata["DifficultyRating"] < self.args.min_difficulty or
                        beatmap_metadata["DifficultyRating"] > self.args.max_difficulty):
                    continue

                for sample in self._get_next_beatmap(audio_samples, i, beatmap_metadata, metadata, speed):
                    yield sample

    def _get_next_beatmap(self, audio_samples, i, beatmap_metadata: Series, set_metadata: DataFrame,
                          speed: float) -> dict:
        context_info = None
        if len(self.args.context_types) > 0:
            # Randomly select a context type with probabilities of context_weights
            context_info = random.choices(self.args.context_types, weights=self.args.context_weights)[0]

            # It's important to copy the context_info because we will modify it, and we don't want to permanently change the config
            context_info = context_info.copy()

            if ContextType.GD in context_info["in"] and len(set_metadata) <= 1:
                context_info["in"].remove(ContextType.GD)
                if len(context_info["in"]) == 0:
                    context_info["in"].append(ContextType.NONE)

            # Make sure we only generate scroll speed contexts for mania
            # Other gamemodes already model all SVs in the map context
            if beatmap_metadata["ModeInt"] != 3 and ContextType.SV in context_info["out"]:
                context_info["out"].remove(ContextType.SV)

        beatmap_path = self.path / "data" / beatmap_metadata["BeatmapSetFolder"] / beatmap_metadata["BeatmapFile"]
        frames, frame_times = self._get_frames(audio_samples)
        osu_beatmap = Beatmap.from_path(beatmap_path)

        def add_special_data(data, beatmap_metadata, beatmap: Beatmap):
            gamemode = beatmap_metadata["ModeInt"]
            data["gamemode"] = gamemode
            data["beatmap_id"] = beatmap.beatmap_id
            data["beatmap_idx"] = beatmap_metadata["BeatmapIdx"]
            data["difficulty"] = self._get_difficulty(beatmap_metadata, speed)
            data["year"] = beatmap_metadata["SubmittedDate"].year
            data["hitsounded"] = get_hitsounded_status(beatmap)
            data["song_length"] = get_song_length(audio_samples, self.args.sample_rate)
            if gamemode in [0, 2]:
                data["global_sv"] = beatmap.slider_multiplier
                data["circle_size"] = beatmap.circle_size
            if gamemode == 3:
                data["keycount"] = int(beatmap.circle_size)
                data["hold_note_ratio"] = get_hold_note_ratio(beatmap)
            if gamemode in [1, 3]:
                data["scroll_speed_ratio"] = get_scroll_speed_ratio(beatmap)

        def get_context(context: ContextType, identifier, add_type=True):
            data = {"extra": {"context_type": context, "add_type": add_type, "id": identifier + '_' + context.value}}
            if context == ContextType.NONE:
                data["events"], data["event_times"] = [], []
            elif context == ContextType.TIMING:
                data["events"], data["event_times"] = self.parser.parse_timing(osu_beatmap, speed)
            elif context == ContextType.NO_HS:
                hs_events, hs_event_times = self.parser.parse(osu_beatmap, speed)
                data["events"], data["event_times"] = remove_events_of_type(hs_events, hs_event_times,
                                                                            [EventType.HITSOUND, EventType.VOLUME])
            elif context == ContextType.GD:
                other_metadata = set_metadata.drop(i).sample().iloc[0]
                other_beatmap_path = self.path / "data" / other_metadata["BeatmapSetFolder"] / other_metadata[
                    "BeatmapFile"]
                other_beatmap = Beatmap.from_path(other_beatmap_path)
                data["events"], data["event_times"] = self.parser.parse(other_beatmap, speed)
                add_special_data(data["extra"], other_metadata, other_beatmap)
            elif context == ContextType.MAP:
                data["events"], data["event_times"] = self.parser.parse(osu_beatmap, speed)
            elif context == ContextType.KIAI:
                data["events"], data["event_times"] = self.parser.parse_kiai(osu_beatmap, speed)
            elif context == ContextType.SV:
                data["events"], data["event_times"] = self.parser.parse_scroll_speeds(osu_beatmap, speed)
            return data

        extra_data = {
            "beatmap_idx": torch.tensor(beatmap_metadata["BeatmapIdx"]
                                        if self.test or random.random() >= self.args.class_dropout_prob else self.tokenizer.num_classes, dtype=torch.long),
            "mapper_idx": torch.tensor(self.tokenizer.get_mapper_idx(beatmap_metadata["UserId"])
                                       if self.test or random.random() >= self.args.mapper_dropout_prob else self.tokenizer.num_mapper_classes, dtype=torch.long),
            "difficulty": torch.tensor(self._get_difficulty(beatmap_metadata, speed), dtype=torch.float32),
            "special": {},
        }

        add_special_data(extra_data["special"], beatmap_metadata, osu_beatmap)

        if self.sample_weights is not None:
            extra_data["sample_weights"] = self.sample_weights.get(osu_beatmap.beatmap_id, 1.0)

        out_context = [get_context(context, "out", add_type=self.args.add_out_context_types) for context in context_info["out"]]
        in_context = [get_context(context, "in") for context in context_info["in"]]

        if self.args.add_gd_context:
            in_context.append(get_context(ContextType.GD, "extra_gd", False))

        sequences = self._create_sequences(
            frames,
            frame_times,
            out_context,
            in_context,
            extra_data,
        )

        for sequence in sequences:
            self.maybe_change_dataset()
            sequence = self._normalize_time_shifts(sequence, beatmap_path)
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            # noinspection PyUnresolvedReferences
            if not self.add_empty_sequences and ((sequence["labels"] == self.tokenizer.eos_id) | (
                    sequence["labels"] == LABEL_IGNORE_ID)).all():
                continue
            # if sequence["decoder_input_ids"][self.pre_token_len - 1] != self.tokenizer.pad_id:
            #     continue
            yield sequence
