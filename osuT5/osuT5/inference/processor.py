from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from slider import Beatmap, TimingPoint
from tqdm import tqdm

from config import InferenceConfig
from .server import InferenceClient, model_generate
from ..dataset.osu_parser import OsuParser
from ..dataset.data_utils import (update_event_times, remove_events_of_type, get_hold_note_ratio,
                                  get_scroll_speed_ratio, get_hitsounded_status)
from ..model import Mapperatorinator
from ..tokenizer import Event, EventType, Tokenizer, ContextType

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


@dataclass
class GenerationConfig:
    gamemode: int = 0
    beatmap_id: Optional[int] = None
    difficulty: Optional[float] = None
    mapper_id: Optional[int] = None
    year: Optional[int] = None
    hitsounded: bool = True
    hp_drain_rate: Optional[float] = None
    circle_size: Optional[float] = None
    overall_difficulty: Optional[float] = None
    approach_rate: Optional[float] = None
    slider_multiplier: float = 1.4
    slider_tick_rate: Optional[float] = None
    keycount: int = 4
    hold_note_ratio: Optional[float] = None
    scroll_speed_ratio: Optional[float] = None
    descriptors: Optional[list[str]] = None
    negative_descriptors: Optional[list[str]] = None


# noinspection PyProtectedMember
def generation_config_from_beatmap(beatmap: Beatmap, tokenizer: Tokenizer) -> GenerationConfig:
    gamemode = int(beatmap.mode)
    difficulty = None
    if gamemode == 0 and len(beatmap._hit_objects) > 0:  # We don't have diffcalc for other gamemodes
        try:
            difficulty = round(float(beatmap.stars()), 2)
        except Exception:
            pass
    return GenerationConfig(
        gamemode=gamemode,
        beatmap_id=beatmap.beatmap_id,
        difficulty=difficulty,
        mapper_id=tokenizer.beatmap_mapper.get(beatmap.beatmap_id, None),
        hp_drain_rate=beatmap.hp_drain_rate,
        circle_size=beatmap.circle_size,
        overall_difficulty=beatmap.overall_difficulty,
        approach_rate=beatmap.approach_rate,
        slider_multiplier=beatmap.slider_multiplier,
        slider_tick_rate=beatmap.slider_tick_rate,
        hitsounded=get_hitsounded_status(beatmap),
        keycount=int(beatmap.circle_size) if gamemode == 3 else 4,
        hold_note_ratio=get_hold_note_ratio(beatmap) if gamemode == 3 else None,
        scroll_speed_ratio=get_scroll_speed_ratio(beatmap) if gamemode in [1, 3] else None,
        descriptors=[tokenizer.descriptor_name(idx) for idx in tokenizer.beatmap_descriptors.get(beatmap.beatmap_id, [])] if beatmap.beatmap_id in tokenizer.beatmap_descriptors else None,
    )


class Processor(object):
    def __init__(self, args: InferenceConfig, model: Mapperatorinator | InferenceClient, tokenizer: Tokenizer, cfg_scale: float = None):
        """Model inference stage that processes sequences."""
        self.device = args.device
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.tgt_seq_len = args.train.data.tgt_seq_len
        self.frame_seq_len = args.train.data.src_seq_len - 1
        self.frame_size = args.train.model.spectrogram.hop_length
        self.sample_rate = args.train.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.miliseconds_per_sequence = self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        self.lookback_time = args.lookback * self.miliseconds_per_sequence
        self.lookback_time_range = range(tokenizer.event_start[EventType.TIME_SHIFT], tokenizer.encode(Event(EventType.TIME_SHIFT, int(self.lookback_time / MILISECONDS_PER_STEP))))
        self.lookahead_max_time = (1 - args.lookahead) * self.miliseconds_per_sequence
        self.lookahead_time = args.lookahead * self.miliseconds_per_sequence
        self.lookahead_time_range = range(tokenizer.encode(Event(EventType.TIME_SHIFT, int(self.lookahead_max_time / MILISECONDS_PER_STEP))), tokenizer.event_end[EventType.TIME_SHIFT])
        self.eos_time = (1 - args.train.data.lookahead) * self.miliseconds_per_sequence
        self.center_pad_decoder = args.train.data.center_pad_decoder
        # All Special Prefix Tokens
        self.add_out_context_types = args.train.data.add_out_context_types
        self.add_gamemode_token = args.train.data.add_gamemode_token
        self.add_style_token = args.train.data.add_style_token
        self.add_diff_token = args.train.data.add_diff_token
        self.add_mapper_token = args.train.data.add_mapper_token
        self.add_year_token = args.train.data.add_year_token
        self.add_hitsounded_token = args.train.data.add_hitsounded_token
        self.add_song_length_token = args.train.data.add_song_length_token
        self.add_global_sv_token = args.train.data.add_global_sv_token
        self.add_cs_token = args.train.data.add_cs_token
        self.add_keycount_token = args.train.data.add_keycount_token
        self.add_hold_note_ratio_token = args.train.data.add_hold_note_ratio_token
        self.add_scroll_speed_ratio_token = args.train.data.add_scroll_speed_ratio_token
        self.add_descriptors = args.train.data.add_descriptors
        self.add_sv_special_token = args.train.data.add_sv_special_token
        self.add_kiai_special_token = args.train.data.add_kiai_special_token
        self.add_song_position_token = args.train.data.add_song_position_token
        # ---
        self.add_kiai = args.train.data.add_kiai
        self.max_pre_token_len = args.train.data.max_pre_token_len
        self.add_pre_tokens = args.train.data.add_pre_tokens
        self.add_gd_context = args.train.data.add_gd_context
        self.add_timing = args.train.data.add_timing
        self.parser = OsuParser(args.train, self.tokenizer)
        self.do_style_embed = args.train.model.do_style_embed
        self.do_difficulty_embed = args.train.model.do_difficulty_embed
        self.do_mapper_embed = args.train.model.do_mapper_embed
        self.do_song_position_embed = args.train.model.do_song_position_embed
        self.add_positions = args.train.data.add_positions
        self.add_sv = args.train.data.add_sv
        self.add_mania_sv = args.train.data.add_mania_sv
        self.context_types: list[dict[str, list[ContextType]]] = \
            [{k: [ContextType(t) for t in v] for k, v in ct.items()} for ct in args.train.data.context_types]
        self.add_to_beatmap = args.add_to_beatmap
        self.start_time = args.start_time
        self.end_time = args.end_time

        if self.add_positions:
            self.position_precision = args.train.data.position_precision
            x_min, x_max, y_min, y_max = args.train.data.position_range
            self.x_min = x_min // self.position_precision
            self.x_max = x_max // self.position_precision
            self.y_min = y_min // self.position_precision
            self.y_max = y_max // self.position_precision
            self.x_count = self.x_max - self.x_min + 1

        self.cfg_scale = cfg_scale or args.cfg_scale
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.temperature = args.temperature
        self.timing_temperature = args.timing_temperature
        self.mania_column_temperature = args.mania_column_temperature
        self.taiko_hit_temperature = args.taiko_hit_temperature
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.parallel = args.parallel
        self.max_batch_size = args.max_batch_size

        self.timeshift_bias = args.timeshift_bias
        self.types_first = args.train.data.types_first

    def model_generate(self, model_kwargs, **generate_kwargs: Any) -> Any:
        generate_kwargs2 = generate_kwargs | dict(
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            top_p=self.top_p,
            top_k=self.top_k,
            max_length=self.tgt_seq_len,
            cfg_scale=self.cfg_scale,
            timeshift_bias=self.timeshift_bias,
            types_first=self.types_first,
            temperature=self.temperature,
            timing_temperature=self.timing_temperature,
            mania_column_temperature=self.mania_column_temperature,
            taiko_hit_temperature=self.taiko_hit_temperature,
        )

        if isinstance(self.model, InferenceClient):
            return self.model.generate(model_kwargs, generate_kwargs2)
        else:
            return model_generate(self.model, self.tokenizer, model_kwargs, generate_kwargs2)

    def generate(
            self,
            *,
            sequences: tuple[torch.Tensor, torch.Tensor, float],
            generation_config: GenerationConfig,
            in_context: list[ContextType] = None,
            out_context: list[ContextType] = None,
            beatmap_path: Optional[str] = None,
            extra_in_context: Optional[dict[ContextType, tuple[list[Event], list[int]] | tuple[list[Event], list[int], torch.Tensor] | list[TimingPoint]]] = None,
            verbose: bool = True,
    ) -> list[tuple[list[Event], list[int]]]:
        """Generate a list of Event object lists and their timestamps given source sequences.

        Args:
            sequences: A list of batched source sequences, and the total song length in milliseconds.
            generation_config: Generation configuration.
            in_context: List of context information.
            out_context: Output contexts to generate.
            beatmap_path: Path to the beatmap file for context generation.
            extra_in_context: Extra context information to use instead of beatmap_path.
            verbose: Whether to show progress bar.

        Returns:
            events: List of Event object lists.
            event_times: Corresponding event times of Event object lists in miliseconds.
        """
        # Merge extra in context with in context
        if extra_in_context is not None:
            in_context = in_context.copy() if in_context is not None else []
            for context_type in extra_in_context:
                if context_type not in in_context:
                    in_context.append(context_type)

        # Find a viable context generation template
        viable_templates = [
            context_type for context_type in self.context_types if
            all(oc in context_type["out"] for oc in out_context) and all(ic in in_context or ic == ContextType.NONE for ic in context_type["in"])
        ]

        if len(viable_templates) == 0:
            raise ValueError("No viable template found for the given context types. Candidates are: " + str(self.context_types))

        # Use the template with the most non-none in context
        template = max(viable_templates, key=lambda ct: sum(1 for ic in ct["in"] if ic != ContextType.NONE))
        all_out_context = template["out"].copy()

        # Only generate SV in mania mode
        if generation_config.gamemode != 3:
            if ContextType.SV in all_out_context:
                all_out_context.remove(ContextType.SV)
            if ContextType.SV in out_context:
                out_context.remove(ContextType.SV)

        # We have to generate the out contexts in order of the template
        out_context_count = max(all_out_context.index(oc) for oc in out_context) + 1
        out_context_to_generate = all_out_context[:out_context_count]
        req_special_tokens = self.get_required_extra_special_tokens(all_out_context)

        song_length = sequences[2]
        in_context_data = self.get_in_context(
            in_context=template["in"],
            beatmap_path=beatmap_path,
            extra_in_context=extra_in_context,
            song_length=song_length,
        )
        out_context_data = self.get_out_context(
            out_context=out_context_to_generate,
            generation_config=generation_config,
            given_context=in_context,
            beatmap_path=beatmap_path,
            extra_in_context=extra_in_context,
            song_length=song_length,
            verbose=verbose,
        )

        # Prepare special conditioning input for model kwargs
        model_kwargs = {}
        if self.do_style_embed:
            if generation_config.beatmap_id is not None:
                model_kwargs["beatmap_idx"] = torch.tensor(
                    [self.tokenizer.beatmap_idx[generation_config.beatmap_id]], dtype=torch.long)
            else:
                model_kwargs["beatmap_idx"] = torch.tensor([self.tokenizer.num_classes], dtype=torch.long)
        if self.do_difficulty_embed:
            if generation_config.difficulty is not None:
                model_kwargs["difficulty"] = torch.tensor([generation_config.difficulty], dtype=torch.float32)
            else:
                # print("WARNING: Difficulty not provided. Selecting 5.0 for difficulty.")
                model_kwargs["difficulty"] = torch.tensor([5.0], dtype=torch.float32)
        if self.do_mapper_embed:
            if generation_config.mapper_id is not None:
                model_kwargs["mapper_idx"] = torch.tensor([self.tokenizer.get_mapper_idx(generation_config.mapper_id)], dtype=torch.long)
            else:
                # print("WARNING: Mapper ID not provided. Selecting default mapper.")
                model_kwargs["mapper_idx"] = torch.tensor([-1], dtype=torch.long)

        # Start generation
        inputs = dict(
            sequences=sequences,
            in_context=in_context_data,
            out_context=out_context_data,
            model_kwargs=model_kwargs,
            req_special_tokens=req_special_tokens,
            verbose=verbose,
        )

        generate_func = self.generate_parallel if self.parallel else self.generate_sequential
        if isinstance(self.model, InferenceClient):
            with self.model:
                generate_func(**inputs)
        else:
            generate_func(**inputs)

        # Post-process events
        for context in out_context_data:
            # Regenerate event times
            context["event_times"] = []
            update_event_times(context["events"], context["event_times"], song_length, self.types_first)

            # Trim events to start and end time
            # Add extra leniency because generated events may not be exactly on time
            if self.start_time is not None:
                self._trim_events_before_time(context["events"], context["event_times"], self.start_time - 10)
            if self.end_time is not None:
                self._trim_events_after_time(context["events"], context["event_times"], self.end_time + 10)

            # Rescale and unpack position events
            if context["context_type"] == ContextType.MAP and self.add_positions:
                context["events"], context["event_times"] = self._rescale_positions(context["events"], context["event_times"])

        # If we are adding to beatmap, add back the events of the reference beatmap
        if self.add_to_beatmap and (self.start_time is not None or self.end_time is not None):
            parser = OsuParser(self.args.train, self.tokenizer)
            parser.position_precision = 1
            parser.position_split_axes = True
            for context in out_context_data:
                ref_context = self.get_context(
                    context["context_type"],
                    beatmap_path=beatmap_path,
                    extra_in_context=extra_in_context,
                    finished=True,
                    parser=parser,
                )
                if self.start_time is not None:
                    ref_events, ref_event_times = ref_context["events"].copy(), ref_context["event_times"].copy()
                    self._trim_events_after_time(ref_events, ref_event_times, self.start_time - 1)
                    context["events"] = ref_events + context["events"]
                    context["event_times"] = ref_event_times + context["event_times"]
                if self.end_time is not None:
                    ref_events, ref_event_times = ref_context["events"].copy(), ref_context["event_times"].copy()
                    self._trim_events_before_time(ref_events, ref_event_times, self.end_time + 1)
                    context["events"] += ref_events
                    context["event_times"] += ref_event_times

        # Turn mania key column into X position
        for context in out_context_data:
            if context["context_type"] != ContextType.MAP or generation_config.gamemode != 3:
                continue

            context["events"], context["event_times"] = self._convert_column_to_position(context["events"], context["event_times"], generation_config.keycount)

        return [(context["events"], context["event_times"]) for context in out_context_data if context["context_type"] in out_context]

    def generate_sequential(
            self,
            *,
            sequences: tuple[torch.Tensor, torch.Tensor, float],
            in_context: list[dict[str, Any]],
            out_context: list[dict[str, Any]],
            model_kwargs: dict[str, Any],
            req_special_tokens: list[str],
            verbose: bool = True,
    ):
        song_length = sequences[2]

        for i, context in enumerate(out_context):
            if context["finished"]:
                continue

            if verbose:
                print(f"Generating {context['context_type'].value}")
            iterator = tqdm(list(zip(*sequences[:2]))) if verbose else zip(*sequences[:2])
            for sequence_index, (frames, frame_time) in enumerate(iterator):
                trim_lookback = sequence_index != 0 and self.types_first and self.lookback_time > 0
                trim_lookahead = sequence_index != len(sequences[0]) - 1

                # noinspection PyUnresolvedReferences
                frames = self.prepare_frames(frames)
                frame_time = frame_time.item()

                # Get relevant tokens for current frame
                cond_prompt, uncond_prompt = self.get_prompts(
                    self.prepare_context_sequences(in_context, frame_time, False, req_special_tokens),
                    self.prepare_context_sequences(out_context[:i + 1], frame_time, True, req_special_tokens),
                )

                [prompt, uncond_prompt], max_len = self.pad_prompts([cond_prompt, uncond_prompt])

                # Prepare additional model kwargs
                if self.do_song_position_embed:
                    global_pos_start = frame_time / song_length
                    global_pos_end = (frame_time + self.miliseconds_per_sequence) / song_length
                    model_kwargs["song_position"] = torch.tensor([global_pos_start, global_pos_end], dtype=torch.float32).unsqueeze(0)

                result = self.model_generate(
                    model_kwargs | dict(
                        inputs=frames,
                        decoder_input_ids=prompt,
                        decoder_attention_mask=prompt.ne(self.tokenizer.pad_id),
                        negative_prompt=uncond_prompt,
                        negative_prompt_attention_mask=uncond_prompt.ne(self.tokenizer.pad_id) if uncond_prompt is not None else None,
                    ),
                    lookback_time=self.lookback_time if trim_lookback else 0,
                    lookahead_time=self.lookahead_time if trim_lookahead else 0,
                    context_type=context["context_type"].value,
                )

                # Only support batch size 1
                predicted_tokens = result[0, max_len:].cpu()
                self.add_predicted_tokens_to_context(context, predicted_tokens, frame_time, trim_lookback, trim_lookahead)

    def generate_parallel(
            self,
            *,
            sequences: tuple[torch.Tensor, torch.Tensor, float],
            in_context: list[dict[str, Any]],
            out_context: list[dict[str, Any]],
            model_kwargs: dict[str, Any],
            req_special_tokens: list[str],
            verbose: bool = True,
    ):
        # Get relevant inputs
        frames = self.prepare_frames(sequences[0])
        frame_times = sequences[1]
        song_length = sequences[2]
        cond_prompts = []
        uncond_prompts = []
        model_kwargses = []

        for i in range(len(frames)):
            frame_time = frame_times[i].item()
            cond_prompt, uncond_prompt = self.get_prompts(
                self.prepare_context_sequences(in_context, frame_time, False, req_special_tokens),
                self.prepare_context_sequences(out_context[:1], frame_time, True, req_special_tokens),
            )
            cond_prompts.append(cond_prompt)
            uncond_prompts.append(uncond_prompt)

            kwargs = model_kwargs.copy()
            # Prepare additional model kwargs
            if self.do_song_position_embed:
                global_pos_start = frame_time / song_length
                global_pos_end = (frame_time + self.miliseconds_per_sequence) / song_length
                kwargs["song_position"] = torch.tensor([global_pos_start, global_pos_end], dtype=torch.float32).unsqueeze(0)
            model_kwargses.append(kwargs)

        prompt, uncond_prompt, max_len = self.stack_prompts(cond_prompts, uncond_prompts)

        # Split prompts and uncond_prompt into batches
        max_batch_size = max(1, self.max_batch_size // self.num_beams // (2 if self.cfg_scale > 1 else 1))
        num_samples = prompt.size(0)
        model_kwarg_keys = list(model_kwargses[0].keys())
        results = []

        # Process each batch
        iterator = tqdm(list(range(0, num_samples, max_batch_size))) if verbose else range(0, num_samples, max_batch_size)
        for i in iterator:
            frames_batch = frames[i:i + max_batch_size]
            prompt_batch = prompt[i:i + max_batch_size]
            uncond_prompt_batch = uncond_prompt[i:i + max_batch_size] if uncond_prompt is not None else None
            kwargses_batch = model_kwargses[i:i + max_batch_size]
            model_kwargs_batch = {k: torch.cat([kwargs[k] for kwargs in kwargses_batch], dim=0) for k in model_kwarg_keys}

            # Start generation
            results.append(self.model_generate(
                model_kwargs_batch | dict(
                    inputs=frames_batch,
                    decoder_input_ids=prompt_batch,
                    decoder_attention_mask=prompt_batch.ne(self.tokenizer.pad_id),
                    negative_prompt=uncond_prompt_batch,
                    negative_prompt_attention_mask=uncond_prompt_batch.ne(self.tokenizer.pad_id) if uncond_prompt_batch is not None else None,
                ),
            ))

        # Concatenate all batch results to form the final result
        padded_results, _ = self.pad_prompts(results)
        result = torch.cat(padded_results, dim=0)

        predicted_tokens = result[:len(cond_prompts), max_len:].cpu()
        for i in range(len(predicted_tokens)):
            frame_time = frame_times[i].item()
            if self.add_out_context_types:
                for context in out_context:
                    # Find the tokens in predicted_tokens[i] between context sos and eos
                    sos = self.tokenizer.context_sos[context["context_type"]]
                    eos = self.tokenizer.context_eos[context["context_type"]]
                    start = (predicted_tokens[i] == sos).nonzero(as_tuple=True)[0]
                    start = start[0] if len(start) > 0 else 0
                    end = (predicted_tokens[i] == eos).nonzero(as_tuple=True)[0]
                    end = end[0] if len(end) > 0 else len(predicted_tokens[i])
                    self.add_predicted_tokens_to_context(context, predicted_tokens[i, start:end], frame_time)
            else:
                self.add_predicted_tokens_to_context(out_context[0], predicted_tokens[i], frame_time)

    def split_into_batches(self, tensor, max_batch_size, batch_size=1):
        if tensor is None:
            return [None] * batch_size
        return [tensor[i:i + max_batch_size] for i in range(0, tensor.size(0), max_batch_size)]

    def prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 1:
            frames = frames.unsqueeze(0)

        return frames

    def pad_prompts(self, prompts):
        max_len = max(tensor.size(1) if tensor is not None else 0 for tensor in prompts)
        prompts = [torch.nn.functional.pad(tensor, (max_len - tensor.size(1), 0)) if tensor is not None else None for tensor in prompts]
        return prompts, max_len

    def stack_prompts(self, cond_prompts, uncond_prompts):
        length = len(cond_prompts)
        padded_prompts, max_len = self.pad_prompts(cond_prompts + uncond_prompts)
        cond_prompt = torch.cat(padded_prompts[:length], dim=0)

        if self.cfg_scale > 1:
            uncond_prompt = torch.cat(padded_prompts[length:], dim=0)
        else:
            uncond_prompt = None

        return cond_prompt, uncond_prompt, max_len

    def get_context(
            self,
            context: ContextType,
            *,
            beatmap_path: Optional[str] = None,
            extra_in_context: Optional[dict[ContextType, tuple[list[Event], list[int]] | tuple[list[Event], list[int], torch.Tensor] | list[TimingPoint]]] = None,
            song_length: Optional[float] = None,
            add_type: bool = False,
            add_class: bool = False,
            finished: bool = False,
            partial: bool = False,
            parser: Optional[OsuParser] = None,
    ):
        if context != ContextType.NONE and finished and context not in extra_in_context:
            beatmap_path = Path(beatmap_path)
            if not beatmap_path.is_file():
                raise FileNotFoundError(f"Beatmap file {beatmap_path} not found.")

        data = {
            "events": [],
            "event_times": [],
            "context_type": context,
            "add_type": add_type,
            "add_class": add_class,
            "add_pre_tokens": False,
            "song_length": song_length,
            "finished": finished,
        }

        if finished or partial:
            parser = parser or self.parser

            if extra_in_context is not None and context in extra_in_context:
                if context == ContextType.TIMING and isinstance(extra_in_context[context], list):
                    # This is a list of timingpoints
                    timing = extra_in_context[context]
                    data["events"], data["event_times"] = parser.parse_timing(timing, song_length=song_length)
                else:
                    if len(extra_in_context[context]) == 2:
                        data["events"], data["event_times"] = extra_in_context[context]
                    elif len(extra_in_context[context]) == 3:
                        data["events"], data["event_times"], data["class"] = extra_in_context[context]
            elif context == ContextType.NONE:
                pass
            elif context == ContextType.TIMING:
                beatmap = Beatmap.from_path(beatmap_path)
                data["events"], data["event_times"] = parser.parse_timing(beatmap, song_length=song_length)
            elif context == ContextType.MAP:
                beatmap = Beatmap.from_path(beatmap_path)
                data["events"], data["event_times"] = parser.parse(beatmap)
                if add_class:
                    data["class"] = self.get_class_vector(generation_config_from_beatmap(beatmap, self.tokenizer), song_length)
            elif context == ContextType.NO_HS:
                beatmap = Beatmap.from_path(beatmap_path)
                hs_events, hs_event_times = parser.parse(beatmap)
                data["events"], data["event_times"] = remove_events_of_type(hs_events, hs_event_times,
                                                                            [EventType.HITSOUND, EventType.VOLUME])
            elif context == ContextType.GD:
                beatmap = Beatmap.from_path(beatmap_path)
                data["events"], data["event_times"] = parser.parse(beatmap)
                if add_class:
                    data["class"] = self.get_class_vector(generation_config_from_beatmap(beatmap, self.tokenizer), song_length)
            elif context == ContextType.KIAI:
                beatmap = Beatmap.from_path(beatmap_path)
                data["events"], data["event_times"] = parser.parse_kiai(beatmap)
            elif context == ContextType.SV:
                beatmap = Beatmap.from_path(beatmap_path)
                data["events"], data["event_times"] = parser.parse_scroll_speeds(beatmap)
            else:
                raise ValueError(f"Invalid context type {context}")

            if not finished and partial:
                self._trim_events_after_time(data["events"], data["event_times"], self.start_time - 1)
        return data

    def get_in_context(
            self,
            *,
            in_context: list[ContextType],
            beatmap_path: Optional[str],
            extra_in_context: Optional[dict[ContextType, tuple[list[Event], list[int]] | tuple[list[Event], list[int], torch.Tensor] | list[TimingPoint]]] = None,
            song_length: float,
    ) -> list[dict[str, Any]]:
        in_context = [self.get_context(
            context,
            beatmap_path=beatmap_path,
            extra_in_context=extra_in_context,
            song_length=song_length,
            add_type=True,
            add_class=True,
            finished=True,
        ) for context in in_context]
        if self.add_gd_context:
            in_context.append(self.get_context(
                ContextType.GD,
                beatmap_path=beatmap_path,
                extra_in_context=extra_in_context,
                song_length=song_length,
                add_type=False,
                add_class=True,
                finished=True,
            ))
        return in_context

    def get_out_context(
            self,
            *,
            out_context: list[ContextType],
            generation_config: GenerationConfig,
            given_context: list[ContextType],
            beatmap_path: Optional[str],
            extra_in_context: Optional[dict[ContextType, tuple[list[Event], list[int]] | tuple[list[Event], list[int], torch.Tensor] | list[TimingPoint]]] = None,
            song_length: float,
            verbose: bool = True
    ):
        out = []
        for i, context in enumerate(out_context):
            context_data = self.get_context(
                context,
                beatmap_path=beatmap_path,
                extra_in_context=extra_in_context,
                song_length=song_length,
                add_type=self.add_out_context_types,
                add_class=False,
                finished=context in given_context,
                partial=self.add_to_beatmap and self.start_time is not None,
            )

            # Add class vector to the first out context
            if i == 0:
                context_data["class"] = self.get_class_vector(generation_config, song_length, verbose=verbose)
                context_data["negative_class"] = self.get_class_vector(GenerationConfig(
                    gamemode=generation_config.gamemode,
                    difficulty=generation_config.difficulty,
                    mapper_id=generation_config.mapper_id if (generation_config.descriptors and len(generation_config.descriptors) > 0) or (generation_config.negative_descriptors and len(generation_config.negative_descriptors) > 0) else None,
                    year=generation_config.year,
                    hp_drain_rate=generation_config.hp_drain_rate,
                    circle_size=generation_config.circle_size,
                    overall_difficulty=generation_config.overall_difficulty,
                    approach_rate=generation_config.approach_rate,
                    slider_multiplier=generation_config.slider_multiplier,
                    slider_tick_rate=generation_config.slider_tick_rate,
                    hitsounded=generation_config.hitsounded,
                    keycount=generation_config.keycount,
                    hold_note_ratio=generation_config.hold_note_ratio,
                    scroll_speed_ratio=generation_config.scroll_speed_ratio,
                    descriptors=generation_config.negative_descriptors,
                ), song_length)
                context_data["add_pre_tokens"] = self.add_pre_tokens

            out.append(context_data)
        return out

    def get_class_vector(
            self,
            config: GenerationConfig,
            song_length: float,
            verbose: bool = False,
    ):
        cond_tokens = []

        if self.add_gamemode_token:
            gamemode_token = self.tokenizer.encode_gamemode(config.gamemode)
            cond_tokens.append(gamemode_token)
        if self.add_style_token:
            style_token = self.tokenizer.encode_style(config.beatmap_id) if config.beatmap_id is not None else self.tokenizer.style_unk
            cond_tokens.append(style_token)
            if config.beatmap_id is not None and config.beatmap_id not in self.tokenizer.beatmap_idx and verbose:
                print(f"Beatmap class {config.beatmap_id} not found. Using default.")
        if self.add_diff_token:
            diff_token = self.tokenizer.encode_diff(config.difficulty) if config.difficulty is not None else self.tokenizer.diff_unk
            cond_tokens.append(diff_token)
        if self.add_mapper_token:
            mapper_token = self.tokenizer.encode_mapper_id(config.mapper_id) if config.mapper_id is not None else self.tokenizer.mapper_unk
            cond_tokens.append(mapper_token)
            if config.mapper_id is not None and config.mapper_id not in self.tokenizer.mapper_idx and verbose:
                print(f"Mapper class {config.mapper_id} not found. Using default.")
        if self.add_year_token:
            year_token = self.tokenizer.encode_year(config.year) if config.year is not None else self.tokenizer.year_unk
            cond_tokens.append(year_token)
        if self.add_hitsounded_token:
            hitsounded_token = self.tokenizer.encode(Event(EventType.HITSOUNDED, int(config.hitsounded)))
            cond_tokens.append(hitsounded_token)
        if self.add_song_length_token:
            song_length_token = self.tokenizer.encode_song_length(song_length)
            cond_tokens.append(song_length_token)
        if self.add_global_sv_token and self.add_sv and config.gamemode in [0, 2]:
            global_sv_token = self.tokenizer.encode_global_sv(config.slider_multiplier)
            cond_tokens.append(global_sv_token)
        if self.add_cs_token and config.gamemode in [0, 2]:
            cs_token = self.tokenizer.encode_cs(config.circle_size) if config.circle_size is not None else self.tokenizer.cs_unk
            cond_tokens.append(cs_token)
        if config.gamemode == 3:
            if self.add_keycount_token:
                keycount_token = self.tokenizer.encode(Event(EventType.MANIA_KEYCOUNT, config.keycount))
                cond_tokens.append(keycount_token)
            if self.add_hold_note_ratio_token:
                hold_note_ratio_token = self.tokenizer.encode_hold_note_ratio(config.hold_note_ratio) if config.hold_note_ratio is not None else self.tokenizer.hold_note_ratio_unk
                cond_tokens.append(hold_note_ratio_token)
        if self.add_scroll_speed_ratio_token and config.gamemode in [1, 3]:
            scroll_speed_ratio_token = self.tokenizer.encode_scroll_speed_ratio(config.scroll_speed_ratio) if config.scroll_speed_ratio is not None else self.tokenizer.scroll_speed_ratio_unk
            cond_tokens.append(scroll_speed_ratio_token)

        descriptors = config.descriptors if config.descriptors is not None else []
        descriptors_added = 0
        if self.add_descriptors:
            if descriptors is not None and len(descriptors) > 0:
                for descriptor in descriptors:
                    if isinstance(descriptor, str):
                        if descriptor not in self.tokenizer.descriptor_idx:
                            if verbose:
                                print(f"Descriptor class {descriptor} not found. Skipping.")
                            continue
                        cond_tokens.append(self.tokenizer.encode_descriptor_name(descriptor))
                        descriptors_added += 1
                    elif isinstance(descriptor, int):
                        if descriptor < self.tokenizer.event_range[EventType.DESCRIPTOR].min_value or \
                                descriptor > self.tokenizer.event_range[EventType.DESCRIPTOR].max_value:
                            if verbose:
                                print(f"Descriptor idx {descriptor} out of range. Skipping.")
                            continue
                        cond_tokens.append(self.tokenizer.encode_descriptor_idx(descriptor))
                        descriptors_added += 1
            if descriptors is None or descriptors_added == 0:
                cond_tokens.append(self.tokenizer.descriptor_unk)

        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long).unsqueeze(0)

        return cond_tokens

    def add_predicted_tokens_to_context(
            self,
            context: dict,
            predicted_tokens,
            frame_time,
            trim_lookback: bool = False,
            trim_lookahead: bool = False
    ):
        # Trim prompt and eos tokens
        while len(predicted_tokens) > 0 and (
                predicted_tokens[-1] == self.tokenizer.eos_id or
                (context["context_type"] in self.tokenizer.context_eos and
                 predicted_tokens[-1] == self.tokenizer.context_eos[context["context_type"]])):
            predicted_tokens = predicted_tokens[:-1]

        if len(predicted_tokens) > 0 and ((trim_lookahead and predicted_tokens[-1] in self.lookahead_time_range) or
                                          (trim_lookback and predicted_tokens[-1] in self.lookback_time_range)):
            # If the type token comes before the timeshift token we should remove the type token too
            if self.types_first:
                predicted_tokens = predicted_tokens[:-2]
            else:
                predicted_tokens = predicted_tokens[:-1]

        result = self._decode(predicted_tokens, frame_time)
        context["events"] += result
        update_event_times(context["events"], context["event_times"], frame_time + self.eos_time, self.types_first)

        # Trim events which are in the lookahead window
        if trim_lookahead:
            lookahead_time = frame_time + self.lookahead_max_time
            self._trim_events_after_time(context["events"], context["event_times"], lookahead_time)

    def get_required_extra_special_tokens(self, all_out_context: list[ContextType]) -> list[str]:
        result = []
        if ContextType.KIAI in all_out_context or (self.add_kiai and any(c in all_out_context for c in [ContextType.GD, ContextType.MAP])):
            result.append("last_kiai")
        if ContextType.SV in all_out_context or ((self.add_sv or self.add_mania_sv) and any(c in all_out_context for c in [ContextType.GD, ContextType.MAP])):
            result.append("last_sv")
        if self.add_song_position_token:
            result.append("song_position")
        return result

    def prepare_context_sequences(self, contexts: list[dict], frame_time, out_context: bool, req_special_tokens: list[str]) -> list[dict]:
        results = []
        for i, context in enumerate(contexts):
            result = self.prepare_context_sequence(context, frame_time)
            results.append(result)
            # Extra special tokens are to be stored in the first output context
            if out_context and i != 0:
                for k, v in result["extra_special_events"].items():
                    results[0]["extra_special_events"][k] = v
                del result["extra_special_events"]

        # Make sure the output context has the required special tokens
        if out_context:
            for k in req_special_tokens:
                if k not in results[0]["extra_special_events"]:
                    results[0]["extra_special_events"][k] = self._default_special_event(k)

        # Tokenize extra special tokens in the correct order
        special_token_order = ["last_kiai", "last_sv", "song_position"]
        for result in results:
            if "extra_special_events" not in result:
                continue
            extra_special_events = result["extra_special_events"]
            extra_special_events = [extra_special_events[k] for k in special_token_order if k in extra_special_events]
            result["extra_special_tokens"] = self._encode(extra_special_events, frame_time)

        return results

    def prepare_context_sequence(self, context: dict, frame_time) -> dict:
        result = context.copy()
        result["frame_time"] = frame_time

        if context["add_pre_tokens"]:
            context_pre_events = self._get_events_time_range(
                context["events"], context["event_times"],
                frame_time - self.miliseconds_per_sequence, frame_time)
            pre_tokens = self._encode(context_pre_events, frame_time)
            if 0 <= self.max_pre_token_len < pre_tokens.shape[1]:
                pre_tokens = pre_tokens[:, -self.max_pre_token_len:]
            result["pre_tokens"] = pre_tokens

        context_events = self._get_events_time_range(
            context["events"], context["event_times"], frame_time, frame_time + self.miliseconds_per_sequence)
        result["tokens"] = self._encode(context_events, frame_time)

        # Prepare extra special tokens
        extra_special_events = {}
        if self.add_kiai_special_token and (context["context_type"] == ContextType.KIAI or (self.add_kiai and context["context_type"] in [ContextType.GD, ContextType.MAP])):
            extra_special_events["last_kiai"] = self._kiai_before_time(context["events"], context["event_times"], frame_time)
        if self.add_sv_special_token and (context["context_type"] == ContextType.SV or ((self.add_sv or self.add_mania_sv) and context["context_type"] in [ContextType.GD, ContextType.MAP])):
            extra_special_events["last_sv"] = self._sv_before_time(context["events"], context["event_times"], frame_time)
        if self.add_song_position_token and "class" in context:
            extra_special_events["song_position"] = self.tokenizer.encode_song_position_event(frame_time, context["song_length"])

        result["extra_special_events"] = extra_special_events

        return result

    # Prepare context type indicator tokens
    def get_context_tokens(self, context, max_token_length=None, add_type_end=True):
        context_type = context["context_type"]
        tokens = context["tokens"]

        # Trim tokens if they are too long
        if max_token_length is not None and tokens.shape[1] > max_token_length:
            tokens = tokens[:, -max_token_length:]

        to_concat = []
        if context["add_type"]:
            to_concat.append(torch.tensor([[self.tokenizer.context_sos[context_type]]], dtype=torch.long))

        if context["add_class"]:
            if "class" in context:
                to_concat.append(context["class"])
            if "extra_special_tokens" in context:
                to_concat.append(context["extra_special_tokens"])

        to_concat.append(tokens)

        if context["add_type"] and add_type_end:
            to_concat.append(torch.tensor([[self.tokenizer.context_eos[context_type]]], dtype=torch.long))

        return torch.concatenate(to_concat, dim=-1)

    def get_prompt(self, in_context, out_context, negative=False, max_token_length=None):
        class_container = out_context[0]
        user_prompt = class_container["negative_class"] if negative else class_container["class"]
        extra_special_tokens = class_container["extra_special_tokens"] if "extra_special_tokens" in class_container else torch.tensor([[]], dtype=torch.long)
        pre_tokens = class_container["pre_tokens"] if "pre_tokens" in class_container else torch.tensor([[]], dtype=torch.long)

        in_tokens = [self.get_context_tokens(context, max_token_length) for context in in_context]
        # We must not add the type end token to the last context because it should be generated by the model
        out_tokens = [self.get_context_tokens(context, max_token_length, i != len(out_context) - 1) for i, context in enumerate(out_context)]

        if max_token_length is not None:
            pre_tokens = pre_tokens[:, -max_token_length:]

        to_concat = in_tokens + [user_prompt, extra_special_tokens, pre_tokens]
        prefix = torch.concatenate(to_concat, dim=-1)

        if self.center_pad_decoder:
            prefix = F.pad(prefix, (self.tgt_seq_len // 2 - prefix.shape[1], 0), value=self.tokenizer.pad_id)

        sos = torch.tensor([[self.tokenizer.sos_id]], dtype=torch.long)
        prompt = torch.concatenate([prefix, sos] + out_tokens, dim=-1)
        return prompt

    def get_prompts(self, in_context, out_context):
        # Prepare classifier-free guidance
        cond_prompt = self.get_prompt(in_context, out_context)
        uncond_prompt = self.get_prompt(in_context, out_context, negative=True) if self.cfg_scale > 1 else None

        # Make sure the prompt is not too long
        i = 0
        max_length = self.tgt_seq_len
        while cond_prompt.shape[1] >= self.tgt_seq_len:
            i += 1
            if i > 10:
                raise ValueError("Prompt is too long.")
            max_length = max_length // 2
            cond_prompt = self.get_prompt(in_context, out_context, max_token_length=max_length)
            uncond_prompt = self.get_prompt(in_context, out_context, negative=True,
                                            max_token_length=max_length) if self.cfg_scale > 1 else None

        return cond_prompt, uncond_prompt

    def _get_events_time_range(self, events: list[Event], event_times: list[float], start_time: float, end_time: float):
        # Look from the end of the list
        s = 0
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] < start_time:
                s = i + 1
                break
        e = 0
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] < end_time:
                e = i + 1
                break
        return events[s:e]

    def _trim_events_before_time(self, events, event_times, time):
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] < time:
                del events[i]
                del event_times[i]

    def _trim_events_after_time(self, events, event_times, time):
        for i in range(len(event_times) - 1, -1, -1):
            if event_times[i] > time:
                del events[i]
                del event_times[i]
            else:
                break

    def _encode(self, events: list[Event], frame_time: float) -> torch.Tensor:
        tokens = torch.empty((1, len(events)), dtype=torch.long)
        timeshift_range = self.tokenizer.event_range[EventType.TIME_SHIFT]
        for i, event in enumerate(events):
            if event.type == EventType.TIME_SHIFT:
                value = int((event.value - frame_time) / MILISECONDS_PER_STEP)
                value = np.clip(value, timeshift_range.min_value, timeshift_range.max_value)
                event = Event(type=event.type, value=value)
            tokens[0, i] = self.tokenizer.encode(event)
        return tokens

    def _decode(self, tokens: torch.Tensor, frame_time: float) -> list[Event]:
        """Converts a list of tokens into Event objects and converts to absolute time values.

        Args:
            tokens: List of tokens.
            frame time: Start time of current source sequence.

        Returns:
            events: List of Event objects.
        """
        events = []
        for token in tokens:
            if token == self.tokenizer.eos_id:
                break

            try:
                event = self.tokenizer.decode(token.item())
            except:
                continue

            if event.type == EventType.TIME_SHIFT:
                event.value = frame_time + event.value * MILISECONDS_PER_STEP

            events.append(event)

        return events

    def _rescale_positions(self, events: list[Event], event_times: list[int]) -> tuple[list[Event], list[int]]:
        new_events = []
        new_event_times = []
        offset = self.position_precision // 2 if self.position_precision > 1 else 0
        for i, event in enumerate(events):
            if event.type == EventType.POS_X or event.type == EventType.POS_Y:
                new_events.append(Event(type=event.type, value=event.value * self.position_precision))
                new_event_times.append(event_times[i])
            elif event.type == EventType.POS:
                new_events.append(Event(type=EventType.POS_X, value=((event.value % self.x_count) + self.x_min) * self.position_precision + offset))
                new_events.append(Event(type=EventType.POS_Y, value=((event.value // self.x_count) + self.y_min) * self.position_precision + offset))
                new_event_times.append(event_times[i])
                new_event_times.append(event_times[i])
            else:
                new_events.append(event)
                new_event_times.append(event_times[i])

        return new_events, new_event_times

    def _kiai_before_time(self, events, event_times, time) -> Event:
        for i in range(len(events) - 1, -1, -1):
            if events[i].type == EventType.KIAI and event_times[i] < time:
                return events[i]
        return self._default_special_event("last_kiai")

    def _sv_before_time(self, events, event_times, time) -> Event:
        for i in range(len(events) - 1, -1, -1):
            if events[i].type == EventType.SCROLL_SPEED and event_times[i] < time:
                return events[i]
        return self._default_special_event("last_sv")

    def _default_special_event(self, name: str) -> Event:
        if name == "last_kiai":
            return Event(EventType.KIAI, 0)
        if name == "last_sv":
            return Event(EventType.SCROLL_SPEED, 100)
        raise ValueError(f"Invalid special event name {name}.")

    def _convert_column_to_position(self, events, event_times, key_count) -> tuple[list[Event], list[int]]:
        new_events = []
        new_event_times = []
        for i, event in enumerate(events):
            if event.type == EventType.MANIA_COLUMN:
                x = int((event.value + 0.5) * 512 / key_count)
                new_events.append(Event(EventType.POS_X, x))
                new_events.append(Event(EventType.POS_Y, 192))
                new_event_times.append(event_times[i])
                new_event_times.append(event_times[i])
            else:
                new_events.append(event)
                new_event_times.append(event_times[i])
        return new_events, new_event_times
