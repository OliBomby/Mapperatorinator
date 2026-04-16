from __future__ import annotations

import random
from multiprocessing.managers import Namespace
from typing import Optional, Any

from huggingface_hub import list_repo_files
from datasets import load_dataset, Audio
import torch
from slider import Beatmap
from torch.utils.data import IterableDataset

from ..config import DataConfig
from ..dataset.osu_parser import OsuParser
from ..tokenizer import Tokenizer, ContextType, EventType
from .data_utils import (
    SequenceDatasetMixin,
    load_web_audio,
    filter_web_beatmaps,
    get_hold_note_ratio,
    get_scroll_speed_ratio,
    get_hitsounded_status,
    get_song_length,
    get_web_submitted_date,
    remove_events_of_type,
    get_speed_augment,
    get_flip_augment,
    calculate_difficulty,
)


class WebDataset(SequenceDatasetMixin, IterableDataset):
    __slots__ = (
        "args",
        "parser",
        "tokenizer",
        "subset_ids",
        "test",
        "repo_id",
        "files_split",
        "frame_seq_len",
        "min_pre_token_len",
        "pre_token_len",
        "add_pre_tokens",
        "add_empty_sequences",
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
        super().__init__()
        self._validate_args(args)
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.subset_ids = subset_ids
        self.test = test
        self.shared = shared
        self.repo_id = args.test_dataset_path if test else args.train_dataset_path
        dataset_start = args.test_dataset_start if test else args.train_dataset_start
        dataset_end = args.test_dataset_end if test else args.train_dataset_end

        if args.dataset_subset is not None:
            subset_prefix = args.dataset_subset if args.dataset_subset.startswith('/') else args.dataset_subset + "/"
            all_files = [f for f in list_repo_files(self.repo_id, repo_type="dataset") if f.startswith(subset_prefix)]
        else:
            all_files = [f for f in list_repo_files(self.repo_id, repo_type="dataset")]

        all_files.sort()
        self.files_split = all_files[dataset_start:dataset_end]

        self.frame_seq_len = args.src_seq_len - 1
        self.min_pre_token_len = 4
        self.pre_token_len = args.tgt_seq_len // 2
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences

    @staticmethod
    def _validate_args(args: DataConfig):
        if not args.per_track:
            raise ValueError("Web dataset requires per_track to be True")
        if args.only_last_beatmap:
            raise ValueError("Web dataset does not support only_last_beatmap")
        if args.cycle_length > 1:
            raise ValueError("Web dataset does not support cycle_length > 1")

    def __iter__(self):
        streaming = self.args.test_dataset_streaming if self.test else self.args.train_dataset_streaming
        dataset = load_dataset(self.repo_id, data_files=self.files_split, streaming=streaming, split="train")

        if not self.test:
            if streaming:
                dataset = dataset.shuffle(seed=42, buffer_size=100)
                dataset.set_epoch(self.shared.current_epoch)
            else:
                dataset = dataset.shuffle(seed=42 + self.shared.current_epoch)

        dataset = dataset.cast_column("opus", Audio(sampling_rate=self.args.sample_rate, num_channels=1))

        return self._iter_rows(dataset)

    def _iter_rows(self, dataset):
        for row in dataset:
            beatmaps = filter_web_beatmaps(
                (row.get("json") or {}).get("beatmaps") or [],
                subset_ids=self.subset_ids,
                gamemodes=self.args.gamemodes,
                ranked_statuses=self.args.ranked_statuses,
                min_year=self.args.min_year,
                max_year=self.args.max_year,
                min_difficulty=self.args.min_difficulty,
                max_difficulty=self.args.max_difficulty,
            )
            if not beatmaps:
                continue

            parsed_entries = []
            for beatmap_metadata in beatmaps:
                content = beatmap_metadata.get("content")
                if not content:
                    continue
                try:
                    parsed_entries.append({
                        "metadata": beatmap_metadata,
                        "beatmap": Beatmap.parse(content),
                    })
                except Exception as e:
                    beatmap_id = beatmap_metadata.get("beatmap_id", "unknown")
                    print(f"Failed to parse web beatmap {beatmap_id}")
                    print(e)

            if not parsed_entries:
                continue

            if self.args.add_gd_context and len(parsed_entries) <= 1:
                continue

            speed = get_speed_augment(
                self.test,
                self.args.dt_augment_prob,
                self.args.dt_augment_range,
                self.args.dt_augment_sqrt,
            )
            flip = get_flip_augment(
                self.test,
                self.args.flip_horizontal_prob,
                self.args.flip_vertical_prob,
            )

            try:
                audio_samples = load_web_audio(row["opus"], self.args.sample_rate, speed, normalize=self.args.normalize_audio)
            except Exception as e:
                print(f"Failed to decode web audio for sample {row.get('__key__', 'unknown')}")
                print(e)
                continue

            frames, frame_times = self._get_frames(audio_samples)
            for i, entry in enumerate(parsed_entries):
                yield from self._get_next_beatmap(audio_samples, frames, frame_times, parsed_entries, i, entry, speed, flip)

    def _get_context_info(self, set_size: int) -> dict[str, list[ContextType]]:
        context_info = random.choices(self.args.context_types, weights=self.args.context_weights)[0].copy()

        if ContextType.GD in context_info["in"] and set_size <= 1:
            context_info["in"].remove(ContextType.GD)
            if len(context_info["in"]) == 0:
                context_info["in"].append(ContextType.NONE)

        return context_info

    def _get_difficulty(self, beatmap_metadata: dict[str, Any], speed: float = 1.0) -> float:
        if speed != 1.0:
            content = beatmap_metadata.get("content")
            if content:
                difficulty = calculate_difficulty(content=content, speed=speed)
                if difficulty is not None:
                    return difficulty
        difficulty = beatmap_metadata.get("difficultyrating")
        return float(difficulty) if difficulty is not None else 0.0

    def _add_special_data(self, data: dict[str, Any], beatmap_metadata: dict[str, Any], beatmap: Beatmap, audio_samples, speed: float = 1.0) -> None:
        gamemode = int(beatmap_metadata["mode"])
        beatmap_id = int(beatmap_metadata["beatmap_id"])
        submitted_date = get_web_submitted_date(beatmap_metadata)

        data["gamemode"] = gamemode
        data["beatmap_id"] = beatmap_id
        data["beatmap_idx"] = self.tokenizer.beatmap_idx.get(beatmap_id, beatmap_id)
        data["difficulty"] = self._get_difficulty(beatmap_metadata, speed)
        if submitted_date is not None:
            data["year"] = submitted_date.year
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

    def _get_next_beatmap(
            self,
            audio_samples,
            frames,
            frame_times,
            set_entries: list[dict[str, Any]],
            index: int,
            entry: dict[str, Any],
            speed: float = 1.0,
            flip: tuple[bool, bool] = (False, False),
    ):
        beatmap_metadata = entry["metadata"]
        osu_beatmap = entry["beatmap"]
        context_info = self._get_context_info(len(set_entries)) if len(self.args.context_types) > 0 else {"in": [], "out": [ContextType.MAP]}

        def get_context(context: ContextType, identifier, add_type=True):
            data = {"extra": {"context_type": context, "add_type": add_type, "id": identifier + '_' + context.value}}
            if context == ContextType.NONE:
                data["events"], data["event_times"] = [], []
            elif context == ContextType.TIMING:
                data["events"], data["event_times"] = self.parser.parse_timing(osu_beatmap, speed)
            elif context == ContextType.NO_HS:
                hs_events, hs_event_times = self.parser.parse(osu_beatmap, speed, None, flip)
                data["events"], data["event_times"] = remove_events_of_type(hs_events, hs_event_times, [EventType.HITSOUND, EventType.VOLUME])
            elif context == ContextType.GD:
                other_entry = random.choice(set_entries[:index] + set_entries[index + 1:])
                data["events"], data["event_times"] = self.parser.parse(other_entry["beatmap"], speed, None, flip)
                self._add_special_data(data["extra"], other_entry["metadata"], other_entry["beatmap"], audio_samples, speed)
            elif context == ContextType.MAP:
                data["events"], data["event_times"] = self.parser.parse(osu_beatmap, speed, None, flip)
            elif context == ContextType.KIAI:
                data["events"], data["event_times"] = self.parser.parse_kiai(osu_beatmap, speed)
            elif context == ContextType.SV:
                if int(beatmap_metadata["mode"]) == 3:
                    data["events"], data["event_times"] = self.parser.parse_scroll_speeds(osu_beatmap, speed)
                else:
                    data["events"], data["event_times"] = [], []
            return data

        beatmap_id = int(beatmap_metadata["beatmap_id"])
        creator_id = int(beatmap_metadata["creator_id"])
        beatmap_idx = self.tokenizer.beatmap_idx.get(beatmap_id, beatmap_id)
        mapper_idx = self.tokenizer.get_mapper_idx(creator_id)
        extra_data = {
            "beatmap_idx": torch.tensor(
                beatmap_idx if self.test or random.random() >= self.args.class_dropout_prob else self.tokenizer.num_classes,
                dtype=torch.long,
            ),
            "mapper_idx": torch.tensor(
                mapper_idx if self.test or random.random() >= self.args.mapper_dropout_prob else self.tokenizer.num_mapper_classes,
                dtype=torch.long,
            ),
            "difficulty": torch.tensor(self._get_difficulty(beatmap_metadata, speed), dtype=torch.float32),
            "special": {},
        }
        self._add_special_data(extra_data["special"], beatmap_metadata, osu_beatmap, audio_samples, speed)

        out_context = [get_context(context, "out", add_type=self.args.add_out_context_types) for context in context_info["out"]]
        in_context = [get_context(context, "in") for context in context_info["in"]]

        if self.args.add_gd_context:
            in_context.append(get_context(ContextType.GD, "extra_gd", False))

        beatmap_path = f"web://{beatmap_metadata.get('beatmapset_id', 'unknown')}/{beatmap_id}"
        sequences = self._create_sequences(
            frames,
            frame_times,
            out_context,
            in_context,
            extra_data,
        )

        for sequence in self.process_sequences(sequences, beatmap_path):
            yield sequence
