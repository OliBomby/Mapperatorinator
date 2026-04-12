from __future__ import annotations

import os
import random
from multiprocessing.managers import Namespace
from typing import Optional, Callable, Any, Generator
from pathlib import Path

import numpy as np
import torch
from pandas import Series, DataFrame
from slider import Beatmap
from torch.utils.data import IterableDataset

from .data_utils import load_audio_file, remove_events_of_type, get_hold_note_ratio, get_scroll_speed_ratio, \
    get_hitsounded_status, get_song_length, load_mmrs_metadata, filter_mmrs_metadata, SequenceDatasetMixin, \
    get_speed_augment
from .osu_parser import OsuParser
from ..tokenizer import EventType, Tokenizer, ContextType
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
        self.metadata = load_mmrs_metadata(self.path)
        self.subset_ids = subset_ids
        self.sample_weights = self._get_sample_weights(args.sample_weights_path)

    def _validate_args(self, args: DataConfig):
        if not args.per_track:
            raise ValueError("MMRS dataset requires per_track to be True")
        if args.only_last_beatmap:
            raise ValueError("MMRS dataset does not support only_last_beatmap")

    def _get_filtered_metadata(self):
        """Get the subset IDs for the dataset with all filtering applied."""
        return filter_mmrs_metadata(
            self.metadata,
            start=self.start,
            end=self.end,
            subset_ids=self.subset_ids,
            gamemodes=self.args.gamemodes,
            ranked_statuses=self.args.ranked_statuses,
            min_year=self.args.min_year,
            max_year=self.args.max_year,
            min_difficulty=self.args.min_difficulty,
            max_difficulty=self.args.max_difficulty,
        )

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
        filtered_metadata = self._get_filtered_metadata()

        if not self.test:
            subset_ids = filtered_metadata.index.get_level_values(0).unique().to_numpy()
            subset_ids = np.random.permutation(subset_ids)
            filtered_metadata = filtered_metadata.loc[subset_ids]

        if self.args.cycle_length > 1 and not self.test:
            return InterleavingBeatmapDatasetIterable(
                filtered_metadata,
                self._iterable_factory,
                self.args.cycle_length,
            )

        return self._iterable_factory(filtered_metadata).__iter__()

    def _iterable_factory(self, metadata: DataFrame) -> BeatmapDatasetIterable:
        return BeatmapDatasetIterable(
            metadata,
            self.args,
            self.path,
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
            metadata: DataFrame,
            iterable_factory: Callable,
            cycle_length: int,
    ):
        self.workers = [
            iterable_factory(df).__iter__()
            for df in np.array_split(metadata, cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[Any, int]:
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


class BeatmapDatasetIterable(SequenceDatasetMixin):
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
            metadata: DataFrame,
            args: DataConfig,
            path: Path,
            parser: OsuParser,
            tokenizer: Tokenizer,
            test: bool,
            shared: Namespace,
            sample_weights: dict[int, float] = None,
    ):
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
        # let N = |tgt_seq_len|
        # [SOS] token + event_tokens + [EOS] token creates N+1 tokens
        # [SOS] token + event_tokens[:-1] creates N target sequence
        # event_tokens[1:] + [EOS] token creates N label sequence
        self.min_pre_token_len = 4
        self.pre_token_len = args.tgt_seq_len // 2
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences

    def __iter__(self):
        return self._get_next_tracks()

    def _get_difficulty(self, beatmap_metadata: Series, speed: float = 1.0) -> float:
        # StarRating is an array that gives the difficulty for the speeds:
        # 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        # Linearly interpolate between the two closest speeds
        star_ratings = beatmap_metadata["StarRating"]
        speed_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        return np.interp(speed, speed_ratios, star_ratings)  # type: ignore

    def _get_next_tracks(self) -> Generator[dict, None, None]:
        for beatmapset_id in self.metadata.index.get_level_values(0).unique():
            metadata = self.metadata.loc[beatmapset_id]

            if self.args.add_gd_context and len(metadata) <= 1:
                continue

            speed = get_speed_augment(
                self.test,
                self.args.dt_augment_prob,
                self.args.dt_augment_range,
                self.args.dt_augment_sqrt,
            )
            track_path = self.path / "data" / metadata.iloc[0]["BeatmapSetFolder"]
            audio_path = track_path / metadata.iloc[0]["AudioFile"]
            try:
                audio_samples = load_audio_file(audio_path, self.args.sample_rate, speed, self.args.normalize_audio)
            except Exception as e:
                print(f"Failed to load audio file: {audio_path}")
                print(e)
                continue

            for i, beatmap_metadata in metadata.iterrows():
                yield from self._get_next_beatmap(audio_samples, i, beatmap_metadata, metadata, speed)

    def _get_next_beatmap(self, audio_samples, i, beatmap_metadata: Series, set_metadata: DataFrame,
                          speed: float) -> Generator[dict, None, None]:
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
            # if beatmap_metadata["ModeInt"] != 3 and ContextType.SV in context_info["out"]:
            #     context_info["out"].remove(ContextType.SV)

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
                data["scroll_speed_ratio"] = get_scroll_speed_ratio(beatmap, self.args.mania_bpm_normalized_scroll_speed)

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
                if beatmap_metadata["ModeInt"] == 3:
                    data["events"], data["event_times"] = self.parser.parse_scroll_speeds(osu_beatmap, speed)
                else:
                    data["events"], data["event_times"] = [], []
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
            extra_data["sample_weights"] = torch.tensor(
                self.sample_weights.get(osu_beatmap.beatmap_id, 1.0),
                dtype=torch.float32,
            )

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

        yield from self.process_sequences(sequences, beatmap_path)
