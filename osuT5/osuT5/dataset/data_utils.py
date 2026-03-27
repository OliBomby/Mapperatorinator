import dataclasses
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Iterable, Generator

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pydub import AudioSegment

import numpy.typing as npt
from slider import Beatmap, HoldNote, TimingPoint

from ..event import Event, EventType, ContextType

MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100
BEAT_TYPES = [
    EventType.BEAT,
    EventType.MEASURE,
    EventType.TIMING_POINT,
]
TIMING_TYPES = BEAT_TYPES + [EventType.TIME_SHIFT]

TYPE_EVENTS = [
    EventType.CIRCLE,
    EventType.SPINNER,
    EventType.SPINNER_END,
    EventType.SLIDER_HEAD,
    EventType.BEZIER_ANCHOR,
    EventType.PERFECT_ANCHOR,
    EventType.CATMULL_ANCHOR,
    EventType.RED_ANCHOR,
    EventType.LAST_ANCHOR,
    EventType.SLIDER_END,
    EventType.BEAT,
    EventType.MEASURE,
    EventType.TIMING_POINT,
    EventType.KIAI,
    EventType.HOLD_NOTE,
    EventType.HOLD_NOTE_END,
    EventType.DRUMROLL,
    EventType.DRUMROLL_END,
    EventType.DENDEN,
    EventType.DENDEN_END,
    EventType.SCROLL_SPEED_CHANGE,
]

NON_TIMED_EVENTS = [
    EventType.BEZIER_ANCHOR,
    EventType.PERFECT_ANCHOR,
    EventType.CATMULL_ANCHOR,
    EventType.RED_ANCHOR,
]

TIMED_EVENTS = [
    EventType.CIRCLE,
    EventType.SPINNER,
    EventType.SPINNER_END,
    EventType.SLIDER_HEAD,
    EventType.LAST_ANCHOR,
    EventType.SLIDER_END,
    EventType.BEAT,
    EventType.MEASURE,
    EventType.TIMING_POINT,
    EventType.KIAI,
    EventType.HOLD_NOTE,
    EventType.HOLD_NOTE_END,
    EventType.DRUMROLL,
    EventType.DRUMROLL_END,
    EventType.DENDEN,
    EventType.DENDEN_END,
    EventType.SCROLL_SPEED_CHANGE,
]


def load_audio_file(file: str, sample_rate: int, speed: float = 1.0, normalize: bool = True) -> npt.NDArray:
    """Load an audio file as a numpy time-series array

    The signals are resampled, converted to mono channel, and normalized.

    Args:
        file: Path to audio file.
        sample_rate: Sample rate to resample the audio.
        speed: Speed multiplier for the audio.
        normalize: If True, normalize the audio samples to the range [-1, 1].

    Returns:
        samples: Audio time series.
    """
    file = Path(file)
    audio = AudioSegment.from_file(file)
    audio.frame_rate = int(audio.frame_rate * speed)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if normalize:
        samples *= 1.0 / np.max(np.abs(samples))
    return samples


def normalize_audio_samples(samples: npt.ArrayLike) -> npt.NDArray:
    samples = np.asarray(samples, dtype=np.float32)
    peak = np.max(np.abs(samples)) if samples.size > 0 else 0
    if peak > 0:
        samples = samples / peak
    return samples


def decode_web_audio(audio_decoder: Any, normalize: bool = True) -> npt.NDArray:
    samples = audio_decoder.get_all_samples().data
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples[0]
    return normalize_audio_samples(samples) if normalize else samples


def load_mmrs_metadata(path) -> DataFrame:
    # Loads the metadata parquet from the dataset path
    df = pd.read_parquet(Path(path) / "metadata.parquet")
    df["BeatmapIdx"] = df.index
    df.set_index(["BeatmapSetId", "Id"], inplace=True)
    df.sort_index(inplace=True)
    return df


def filter_mmrs_metadata(
        df: DataFrame,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        subset_ids: Optional[list[int]] = None,
        gamemodes: Optional[list[int]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
) -> DataFrame:
    """Filter the MMRs metadata DataFrame based on the given criteria.

    Args:
        df: DataFrame containing the metadata.
        start: Start split index.
        end: End split index.
        subset_ids: List of beatmap IDs to filter by.
        gamemodes: List of gamemodes to filter by.
        min_year: Minimum year to filter by.
        max_year: Maximum year to filter by.
        min_difficulty: Minimum difficulty star rating to filter by.
        max_difficulty: Maximum difficulty star rating to filter by.

    Returns:
        Filtered DataFrame.
    """
    if start is not None and end is not None:
        first_level_labels = df.index.get_level_values(0).unique()
        start_label = first_level_labels[start]
        end_label = first_level_labels[end - 1]
        df = df.loc[start_label:end_label]

    if subset_ids is not None:
        df = df.loc[subset_ids]

    if gamemodes is not None:
        df = df[df["ModeInt"].isin(gamemodes)]

    if min_year is not None:
        df = df[df["RankedDate"] >= datetime(min_year, 1, 1)]

    if max_year is not None:
        df = df[df["RankedDate"] < datetime(max_year + 1, 1, 1)]

    if min_difficulty is not None:
        df = df[df["DifficultyRating"] >= min_difficulty]

    if max_difficulty is not None:
        df = df[df["DifficultyRating"] <= max_difficulty]

    return df


def parse_web_datetime(value: Any) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value
    value = str(value).strip()
    if not value:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def get_web_ranked_date(beatmap_metadata: dict[str, Any]) -> Optional[datetime]:
    return parse_web_datetime(beatmap_metadata.get("approved_date")) or parse_web_datetime(beatmap_metadata.get("submit_date"))


def get_web_submitted_date(beatmap_metadata: dict[str, Any]) -> Optional[datetime]:
    return parse_web_datetime(beatmap_metadata.get("submit_date")) or get_web_ranked_date(beatmap_metadata)


def filter_web_beatmaps(
        beatmaps: Iterable[dict[str, Any]],
        *,
        subset_ids: Optional[list[int]] = None,
        gamemodes: Optional[list[int]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
) -> list[dict[str, Any]]:
    subset_ids_set = set(subset_ids) if subset_ids is not None else None
    filtered = []

    for beatmap in beatmaps:
        beatmapset_id = beatmap.get("beatmapset_id")
        if subset_ids_set is not None and beatmapset_id not in subset_ids_set:
            continue

        mode = beatmap.get("mode")
        if gamemodes is not None and mode not in gamemodes:
            continue

        ranked_date = get_web_ranked_date(beatmap)
        if min_year is not None and (ranked_date is None or ranked_date < datetime(min_year, 1, 1)):
            continue
        if max_year is not None and (ranked_date is None or ranked_date >= datetime(max_year + 1, 1, 1)):
            continue

        difficulty = beatmap.get("difficultyrating")
        if min_difficulty is not None and (difficulty is None or difficulty < min_difficulty):
            continue
        if max_difficulty is not None and (difficulty is None or difficulty > max_difficulty):
            continue

        filtered.append(beatmap)

    return filtered


class SequenceDatasetMixin:
    args: Any
    tokenizer: Any
    test: bool
    shared: Any
    frame_seq_len: int
    min_pre_token_len: int
    pre_token_len: int
    add_pre_tokens: bool
    add_empty_sequences: bool

    def _get_frames(self, samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        samples = np.pad(samples, [0, self.args.hop_length - len(samples) % self.args.hop_length])
        frames = np.reshape(samples, (-1, self.args.hop_length))
        frames_per_milisecond = self.args.sample_rate / self.args.hop_length / MILISECONDS_PER_SECOND
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
        extra_data = {} if extra_data is None else extra_data

        def get_event_indices(events2: list[Event], event_times2: list[int]) -> tuple[list[int], list[int]]:
            start_indices = []
            event_index = 0

            for current_time in frame_times:
                while event_index < len(events2) and event_times2[event_index] < current_time:
                    event_index += 1
                start_indices.append(event_index)

            end_indices = start_indices[1:] + [start_indices[-1]]
            return start_indices, end_indices

        start_indices, end_indices = {}, {}
        for context in in_context + out_context:
            start_indices[context["extra"]["id"]], end_indices[context["extra"]["id"]] = get_event_indices(
                context["events"], context["event_times"])

        sequences = []
        n_frames = len(frames)
        offset = random.randint(0, min(self.frame_seq_len, 2000)) if not self.test and random.random() < self.args.frame_offset_augment_prob else 0
        gen_start_frame_x = int(round(self.args.lookback * self.frame_seq_len)) if not self.test and random.random() < self.args.lookback_prob else 0
        gen_end_frame_x = int(round((1 - self.args.lookahead) * self.frame_seq_len))
        last_kiai = {}
        last_sv = {}

        for frame_start_idx in range(offset, n_frames - gen_start_frame_x, self.frame_seq_len):
            frame_end_idx = min(frame_start_idx + self.frame_seq_len, n_frames)

            gen_start_frame = min(frame_start_idx + gen_start_frame_x, n_frames - 1)
            gen_end_frame = min(frame_start_idx + gen_end_frame_x, n_frames)

            frame_pre_idx = max(frame_start_idx - self.frame_seq_len, 0)

            def slice_events(context, start_idx, end_idx):
                if len(context["events"]) == 0:
                    return []
                identifier = context["extra"]["id"]
                event_start_idx = start_indices[identifier][start_idx]
                event_end_idx = end_indices[identifier][end_idx - 1]
                return context["events"][event_start_idx:event_end_idx]

            def slice_context(context, start_idx, end_idx):
                result = {"events": slice_events(context, start_idx, end_idx)} | context["extra"]
                result["time"] = frame_times[start_idx]
                result["labels_offset"] = start_indices[context["extra"]["id"]][gen_start_frame] - start_indices[context["extra"]["id"]][start_idx]
                return result

            sequence: dict[str, str | int | list[Event] | dict] = {
                "frames": frames[frame_start_idx:frame_end_idx],
                "out_context": [slice_context(context, frame_start_idx, gen_end_frame) for context in out_context],
                "in_context": [slice_context(context, frame_start_idx, frame_end_idx) for context in in_context],
                "song_position": torch.tensor([frame_start_idx / n_frames, frame_end_idx / n_frames], dtype=torch.float32),
            } | extra_data

            sequence["special"] = sequence["special"].copy()
            sequence["special"]["time"] = frame_times[frame_start_idx]

            if out_context and (self.args.add_pre_tokens or self.args.add_pre_tokens_at_step >= 0):
                sequence["pre_events"] = slice_events(out_context[0], frame_pre_idx, frame_start_idx)

            def add_last_kiai(sequence_context, state):
                if (sequence_context["context_type"] != ContextType.KIAI and
                        not (self.args.add_kiai and sequence_context["context_type"] in [ContextType.GD, ContextType.MAP])):
                    return
                sequence_context["last_kiai"] = state.get(sequence_context["id"], Event(EventType.KIAI, 0))
                for event in reversed(sequence_context["events"]):
                    if event.type == EventType.KIAI:
                        state[sequence_context["id"]] = event
                        break

            if self.args.add_kiai_special_token:
                for sequence_context in sequence["in_context"]:
                    add_last_kiai(sequence_context, last_kiai)
                for sequence_context in sequence["out_context"]:
                    add_last_kiai(sequence_context, last_kiai)
                    if "last_kiai" in sequence_context:
                        sequence["special"]["last_kiai"] = sequence_context["last_kiai"]

            def add_last_sv(sequence_context, state):
                if (sequence_context["context_type"] != ContextType.SV and
                        not ((self.args.add_sv or self.args.add_mania_sv) and sequence_context["context_type"] in [ContextType.GD, ContextType.MAP])):
                    return
                sequence_context["last_sv"] = state.get(sequence_context["id"], Event(EventType.SCROLL_SPEED, 100))
                for event in reversed(sequence_context["events"]):
                    if event.type == EventType.SCROLL_SPEED:
                        state[sequence_context["id"]] = event
                        break

            if self.args.add_sv_special_token:
                for sequence_context in sequence["in_context"]:
                    add_last_sv(sequence_context, last_sv)
                for sequence_context in sequence["out_context"]:
                    add_last_sv(sequence_context, last_sv)
                    if "last_sv" in sequence_context:
                        sequence["special"]["last_sv"] = sequence_context["last_sv"]

            sequences.append(sequence)

        return sequences

    def _normalize_time_shifts(self, sequence: dict, beatmap_path) -> dict:
        min_t = self.tokenizer.event_range[EventType.TIME_SHIFT].min_value
        max_t = self.tokenizer.event_range[EventType.TIME_SHIFT].max_value

        def process(events: list[Event], start_time) -> list[Event]:
            for i, event in enumerate(events):
                if event.type == EventType.TIME_SHIFT:
                    t = int((event.value - start_time) * STEPS_PER_MILLISECOND)
                    if t < min_t or t > max_t:
                        print(f"WARNING: Time shift out of range ({t}) in beatmap {beatmap_path}")
                        t = np.clip(t, min_t, max_t)
                    events[i] = Event(EventType.TIME_SHIFT, t)
            return events

        if "pre_events" in sequence and sequence["out_context"]:
            sequence["pre_events"] = process(sequence["pre_events"], sequence["out_context"][0]["time"])

        for context in sequence["in_context"] + sequence["out_context"]:
            context["events"] = process(context["events"], context["time"])

        return sequence

    def _get_special_tokens(self, context: dict) -> list[int]:
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
                                      if self.test or random.random() >= self.args.descriptor_dropout_prob else [self.tokenizer.descriptor_unk])
            if self.args.add_kiai_special_token and "last_kiai" in context:
                special_tokens.append(self.tokenizer.encode(context["last_kiai"]))
            if self.args.add_sv_special_token and "last_sv" in context:
                special_tokens.append(self.tokenizer.encode(context["last_sv"]))
            if self.args.add_song_position_token:
                special_tokens.append(self.tokenizer.encode_song_position(context["time"], context["song_length"]))

        return special_tokens

    def _tokenize_sequence(self, sequence: dict) -> dict:
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
        stl = 1
        stl += len(sequence["special_tokens"])
        for context in sequence["in_context"] + sequence["out_context"]:
            if context["add_type"]:
                stl += 2
            stl += len(context["special_tokens"])

        num_tokens = sum(len(context["tokens"]) for context in sequence["out_context"])
        num_pre_tokens = len(sequence["pre_tokens"]) if "pre_tokens" in sequence else 0
        if self.args.max_pre_token_len > 0:
            num_pre_tokens = min(num_pre_tokens, self.args.max_pre_token_len)
        num_other_tokens = sum(len(context["tokens"]) for context in sequence["in_context"])

        if self.args.center_pad_decoder:
            n = min(self.args.tgt_seq_len - self.pre_token_len - 1, num_tokens)
            m = min(self.pre_token_len - stl + 1, num_pre_tokens)
            o = min(self.pre_token_len - m - stl + 1, num_other_tokens)
            si = self.pre_token_len - m - stl + 1 - o
        else:
            n = min(self.args.tgt_seq_len - stl - min(self.min_pre_token_len, num_pre_tokens), num_tokens)
            m = min(self.args.tgt_seq_len - stl - n, num_pre_tokens)
            o = min(self.args.tgt_seq_len - stl - n - m, num_other_tokens)
            si = 0

        input_tokens = torch.full((self.args.tgt_seq_len,), self.tokenizer.pad_id, dtype=torch.long)
        label_tokens = torch.full((self.args.tgt_seq_len,), LABEL_IGNORE_ID, dtype=torch.long)

        def add_special_tokens(special_tokens, start_index):
            for token in special_tokens:
                input_tokens[start_index] = token
                start_index += 1
            return start_index

        def add_context(context, start_index, max_tokens, add_labels=False):
            if context["add_type"]:
                input_tokens[start_index] = self.tokenizer.context_sos[context["context_type"]]
                if add_labels:
                    label_tokens[start_index - 1] = self.tokenizer.context_sos[context["context_type"]]
                start_index += 1

            start_label_index = start_index + context["labels_offset"]
            start_index = add_special_tokens(context["special_tokens"], start_index)

            num_other_tokens_to_add = min(len(context["tokens"]), max_tokens)
            input_tokens[start_index:start_index + num_other_tokens_to_add] = context["tokens"][:num_other_tokens_to_add]
            start_index += num_other_tokens_to_add
            max_tokens -= num_other_tokens_to_add

            if context["add_type"]:
                input_tokens[start_index] = self.tokenizer.context_eos[context["context_type"]]
                start_index += 1

            if add_labels:
                label_tokens[start_label_index - 1:start_index - 1] = input_tokens[start_label_index:start_index]

            return start_index, max_tokens

        for context in sequence["in_context"]:
            si, o = add_context(context, si, o)

        si = add_special_tokens(sequence["special_tokens"], si)
        start_random_index = si

        if m > 0:
            input_tokens[si:si + m] = sequence["pre_tokens"][-m:]
            si += m

        input_tokens[si] = self.tokenizer.sos_id
        si += 1
        for context in sequence["out_context"]:
            si, n = add_context(context, si, n, True)
        end_index = si
        label_tokens[end_index - 1] = self.tokenizer.eos_id

        def randomize_tokens(tokens):
            offset_tokens = tokens.clone()
            if random.random() < self.args.timing_random_offset_prob:
                offset_tokens += torch.randint(low=-self.args.timing_random_offset, high=self.args.timing_random_offset + 1, size=tokens.shape)
            if random.random() < self.args.timing_random_offset_prob:
                offset_tokens += torch.randint(low=-self.args.timing_random_offset_2, high=self.args.timing_random_offset_2 + 1, size=(1,))
            return torch.where(
                (self.tokenizer.event_start[EventType.TIME_SHIFT] <= tokens) & (tokens < self.tokenizer.event_end[EventType.TIME_SHIFT]),
                torch.clamp(offset_tokens, self.tokenizer.event_start[EventType.TIME_SHIFT], self.tokenizer.event_end[EventType.TIME_SHIFT] - 1),
                tokens,
            )

        if self.args.timing_random_offset > 0 or self.args.timing_random_offset_2 > 0:
            input_tokens[start_random_index:end_index] = randomize_tokens(input_tokens[start_random_index:end_index])

        if self.args.snapping_random_prob > 0:
            random_snappings = torch.randint_like(input_tokens, low=self.tokenizer.event_start[EventType.SNAPPING], high=self.tokenizer.event_end[EventType.SNAPPING])
            mask = (self.tokenizer.event_start[EventType.SNAPPING] <= input_tokens) & (input_tokens < self.tokenizer.event_end[EventType.SNAPPING])
            mask &= torch.rand_like(input_tokens, dtype=torch.float32) < self.args.snapping_random_prob
            input_tokens = torch.where(mask, random_snappings, input_tokens)

        sequence["decoder_input_ids"] = input_tokens
        sequence["decoder_attention_mask"] = input_tokens != self.tokenizer.pad_id
        sequence["labels"] = label_tokens

        del sequence["out_context"]
        del sequence["in_context"]
        del sequence["special_tokens"]
        del sequence["special"]
        if "pre_tokens" in sequence:
            del sequence["pre_tokens"]

        return sequence

    def _pad_frame_sequence(self, sequence: dict) -> dict:
        frames = torch.from_numpy(sequence["frames"]).to(torch.float32)
        if frames.shape[0] != self.frame_seq_len:
            n = min(self.frame_seq_len, len(frames))
            padded_frames = torch.zeros(self.frame_seq_len, frames.shape[-1], dtype=frames.dtype, device=frames.device)
            padded_frames[:n] = frames[:n]
            sequence["frames"] = torch.flatten(padded_frames)
        else:
            sequence["frames"] = torch.flatten(frames)
        return sequence

    def maybe_change_dataset(self):
        shared = getattr(self, "shared", None)
        if shared is None:
            return
        step = shared.current_train_step
        if 0 <= self.args.add_empty_sequences_at_step <= step and not self.add_empty_sequences:
            self.add_empty_sequences = True
        if 0 <= self.args.add_pre_tokens_at_step <= step and not self.add_pre_tokens:
            self.add_pre_tokens = True

    def process_sequences(self, sequences: Iterable[dict], beatmap_path: Any) -> Generator[dict, None, None]:
        for sequence in sequences:
            self.maybe_change_dataset()
            sequence = self._normalize_time_shifts(sequence, beatmap_path)
            sequence = self._tokenize_sequence(sequence)
            sequence = self._pad_frame_sequence(sequence)
            sequence = self._pad_and_split_token_sequence(sequence)
            if not self.add_empty_sequences and ((sequence["labels"] == self.tokenizer.eos_id) | (sequence["labels"] == LABEL_IGNORE_ID)).all():
                continue
            yield sequence


def update_event_times(
        events: list[Event],
        event_times: list[int],
        end_time: Optional[float] = None,
        types_first: bool = False
) -> None:
    """Extends the event times list with the times of the new events if the event list is longer than the event times list.

    Args:
        events: List of events.
        event_times: List of event times.
        end_time: End time of the events, for interpolation.
        types_first: If True, the type token is at the start of the group before the timeshift token.
    """
    start_index = len(event_times)
    end_index = len(events)

    if start_index == end_index:
        return

    current_time = 0 if len(event_times) == 0 else event_times[-1]
    for i in range(start_index, end_index):
        if types_first:
            if i + 1 < end_index and events[i + 1].type == EventType.TIME_SHIFT:
                current_time = events[i + 1].value
        elif events[i].type == EventType.TIME_SHIFT:
            current_time = events[i].value
        event_times.append(current_time)

    # Interpolate time for control point events
    interpolate = False
    if types_first:
        # Start-T-D-CP-D-CP-D-LCP-T-D-End-T-D
        # 1-----1-1-1--1-1--1-7---7-7-9---9-9
        # 1-----1-1-3--3-5--5-7---7-7-9---9-9
        index = range(start_index, end_index)
        current_time = 0 if len(event_times) == 0 else event_times[start_index]
    else:
        # T-D-Start-D-CP-D-CP-T-D-LCP-T-D-End
        # 1-1-1-----1-1--1-1--7-7--7--9-9-9--
        # 1-1-1-----3-3--5-5--7-7--7--9-9-9--
        index = range(end_index - 1, start_index - 1, -1)
        current_time = end_time if end_time is not None else event_times[-1]
    for i in index:
        event = events[i]

        if event.type in TIMED_EVENTS:
            interpolate = False

        if event.type in NON_TIMED_EVENTS:
            interpolate = True

        if not interpolate:
            current_time = event_times[i]
            continue

        if event.type not in NON_TIMED_EVENTS:
            event_times[i] = current_time
            continue

        # Find the time of the first timed event and the number of control points between
        j = i
        step = 1 if types_first else -1
        count = 0
        other_time = current_time
        while 0 <= j < len(events):
            event2 = events[j]
            if event2.type == EventType.TIME_SHIFT:
                other_time = event_times[j]
                break
            if event2.type in NON_TIMED_EVENTS:
                count += 1
            j += step
        if j < 0:
            other_time = 0
        if j >= len(events):
            other_time = end_time if end_time is not None else event_times[-1]

        # Interpolate the time
        current_time = int((current_time - other_time) / (count + 1) * count + other_time)
        event_times[i] = current_time


def merge_events(events1: tuple[list[Event], list[int]], events2: tuple[list[Event], list[int]]) -> tuple[list[Event], list[int]]:
    """Merge two lists of events in a time sorted manner. Assumes both lists are sorted by time.

    Args:
        events1: List of events.
        events2: List of events.

    Returns:
        merged_events: Merged list of events.
        merged_event_times: Merged list of event times.
    """
    merged_events = []
    merged_event_times = []
    i = 0
    j = 0

    while i < len(events1[0]) and j < len(events2[0]):
        t1 = events1[1][i]
        t2 = events2[1][j]

        if t1 <= t2:
            merged_events.append(events1[0][i])
            merged_event_times.append(t1)
            i += 1
        else:
            merged_events.append(events2[0][j])
            merged_event_times.append(t2)
            j += 1

    merged_events.extend(events1[0][i:])
    merged_events.extend(events2[0][j:])
    merged_event_times.extend(events1[1][i:])
    merged_event_times.extend(events2[1][j:])
    return merged_events, merged_event_times


def remove_events_of_type(events: list[Event], event_times: list[int], event_types: list[EventType]) -> tuple[list[Event], list[int]]:
    """Remove all events of a specific type from a list of events.

    Args:
        events: List of events.
        event_times: List of event times.
        event_types: Types of event to remove.

    Returns:
        filtered_events: Filtered list of events.
    """
    new_events = []
    new_event_times = []
    for event, time in zip(events, event_times):
        if event.type not in event_types:
            new_events.append(event)
            new_event_times.append(time)
    return new_events, new_event_times


def events_of_type(events: list[Event], event_times: list[int], event_types: list[EventType]) -> tuple[list[Event], list[int]]:
    """Get all events of a specific type from a list of events.

    Args:
        events: List of events.
        event_times: List of event times.
        event_types: Types of event to keep.

    Returns:
        filtered_events: Filtered list of events.
    """
    new_events = []
    new_event_times = []
    for event, time in zip(events, event_times):
        if event.type in event_types:
            new_events.append(event)
            new_event_times.append(time)
    return new_events, new_event_times


def speed_events(events: tuple[list[Event], list[int]], speed: float) -> tuple[list[Event], list[int]]:
    """Change the speed of a list of events.

    Args:
        events: List of events.
        speed: Speed multiplier.

    Returns:
        sped_events: Sped up list of events.
    """
    sped_events = []
    for event in events[0]:
        if event.type == EventType.TIME_SHIFT:
            event.value = int(event.value / speed)
        sped_events.append(event)

    sped_event_times = []
    for t in events[1]:
        sped_event_times.append(int(t / speed))

    return sped_events, sped_event_times


@dataclasses.dataclass
class Group:
    event_type: EventType = None
    value: int = None
    time: int = 0
    distance: int = None
    x: float = None
    y: float = None
    new_combo: bool = False
    hitsounds: list[int] = dataclasses.field(default_factory=list)
    samplesets: list[int] = dataclasses.field(default_factory=list)
    additions: list[int] = dataclasses.field(default_factory=list)
    volumes: list[int] = dataclasses.field(default_factory=list)
    scroll_speed: float = None


def get_groups(
        events: list[Event],
        *,
        event_times: Optional[list[int]] = None,
        types_first: bool = False
) -> tuple[list[Group], list[list[int]]]:
    groups = []
    group = Group()
    group_indices = []
    indices = []
    for i, event in enumerate(events):
        indices.append(i)
        if event.type == EventType.TIME_SHIFT:
            group.time = event.value
        elif event.type == EventType.DISTANCE:
            group.distance = event.value
        elif event.type == EventType.POS_X:
            group.x = event.value
        elif event.type == EventType.POS_Y:
            group.y = event.value
        elif event.type == EventType.NEW_COMBO:
            group.new_combo = True
        elif event.type == EventType.HITSOUND:
            group.hitsounds.append((event.value % 8) * 2)
            group.samplesets.append(((event.value // 8) % 3) + 1)
            group.additions.append(((event.value // 24) % 3) + 1)
        elif event.type == EventType.VOLUME:
            group.volumes.append(event.value)
        elif event.type == EventType.SCROLL_SPEED:
            group.scroll_speed = event.value / 100
        elif event.type in TYPE_EVENTS:
            if types_first:
                if group.event_type is not None:
                    groups.append(group)
                    group = Group()
                    group_indices.append(indices[:-1])
                    indices = [indices[-1]]
                group.event_type = event.type
                group.value = event.value
                if event_times is not None:
                    group.time = event_times[i]
            else:
                group.event_type = event.type
                group.value = event.value
                if event_times is not None:
                    group.time = event_times[i]
                groups.append(group)
                group = Group()
                group_indices.append(indices)
                indices = []

    if group.event_type is not None:
        groups.append(group)
        group_indices.append(indices)
    elif len(indices) > 0:
        group_indices[-1].extend(indices)

    return groups, group_indices


def get_hold_note_ratio(beatmap: Beatmap) -> Optional[float]:
    notes = beatmap.hit_objects(stacking=False)

    if len(notes) == 0:
        return None

    hold_note_count = 0
    for note in notes:
        if isinstance(note, HoldNote):
            hold_note_count += 1
    return hold_note_count / len(notes)


def get_scroll_speed_ratio(beatmap: Beatmap, mania_normalized: bool = True) -> Optional[float]:
    # Number of scroll speed changes divided by number of distinct hit object times
    notes = beatmap.hit_objects(stacking=False)

    if len(notes) == 0:
        return None

    last_time = -1
    num_note_times = 0
    for note in notes:
        if note.time != last_time:
            num_note_times += 1
            last_time = note.time

    normalized = mania_normalized and beatmap.mode == 3
    median_mpb = get_median_mpb_beatmap(beatmap)
    mpb = median_mpb
    last_normalized_scroll_speed = 1
    num_scroll_speed_changes = 0
    for i, tp in enumerate(beatmap.timing_points):
        if tp.parent is None:
            mpb = tp.ms_per_beat
            scroll_speed = 1
        else:
            scroll_speed = -100 / tp.ms_per_beat

        if i == len(beatmap.timing_points) - 1 or beatmap.timing_points[i + 1].offset > tp.offset:
            normalized_scroll_speed = scroll_speed * median_mpb / mpb if normalized else scroll_speed

            if abs(normalized_scroll_speed - last_normalized_scroll_speed) > 1e-3:
                num_scroll_speed_changes += 1
            last_normalized_scroll_speed = normalized_scroll_speed

    return num_scroll_speed_changes / num_note_times


def get_hitsounded_status(beatmap: Beatmap) -> bool:
    notes = beatmap.hit_objects(stacking=False)
    for note in notes:
        if note.hitsound != 0:
            return True
    return False


def get_song_length(samples: npt.ArrayLike, sample_rate: int) -> float:
    # Length of the audio in milliseconds
    return len(samples) / sample_rate * MILISECONDS_PER_SECOND


def get_median_mpb_beatmap(beatmap: Beatmap) -> float:
    # Not include last slider's end time
    last_time = max(ho.end_time if isinstance(ho, HoldNote) else ho.time for ho in beatmap.hit_objects(stacking=False))
    last_time = int(last_time.seconds * MILISECONDS_PER_SECOND)
    return get_median_mpb(beatmap.timing_points, last_time)


def get_median_mpb(timing_points: list[TimingPoint], last_time: float) -> float:
    # This is identical to osu! stable implementation
    this_beat_length = 0

    bpm_durations = {}

    for i in range(len(timing_points) - 1, -1, -1):
        tp = timing_points[i]
        offset = int(tp.offset.seconds * 1000)

        if tp.parent is None:
            this_beat_length = tp.ms_per_beat

        if this_beat_length == 0 or offset > last_time or (tp.parent is not None and i > 0):
            continue

        if this_beat_length in bpm_durations:
            bpm_durations[this_beat_length] += int(last_time - (0 if i == 0 else offset))
        else:
            bpm_durations[this_beat_length] = int(last_time - (0 if i == 0 else offset))

        last_time = offset

    longest_time = 0
    median = 0

    for bpm, duration in bpm_durations.items():
        if duration > longest_time:
            longest_time = duration
            median = bpm

    return median
