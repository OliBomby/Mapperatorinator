import os.path
from functools import reduce
from pathlib import Path
import random

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf, DictConfig
from slider import Beatmap
from transformers.utils import cached_file

import osu_diffusion
import routed_pickle
from config import InferenceConfig
from diffusion_pipeline import DiffisionPipeline
from osuT5.osuT5.config import TrainConfig
from osuT5.osuT5.dataset.data_utils import events_of_type, TIMING_TYPES, merge_events
from osuT5.osuT5.inference import Preprocessor, Processor, Postprocessor, BeatmapConfig, GenerationConfig, \
    generation_config_from_beatmap, beatmap_config_from_beatmap, background_line
from osuT5.osuT5.inference.super_timing_generator import SuperTimingGenerator
from osuT5.osuT5.model import Mapperatorinator
from osuT5.osuT5.tokenizer import Tokenizer, ContextType
from osuT5.osuT5.utils import get_model
from osu_diffusion import DiT_models
from osu_diffusion.config import DiffusionTrainConfig


def prepare_args(args: InferenceConfig):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 16)
        print(f"Random seed: {args.seed}")
    set_seed(args.seed)


def get_args_from_beatmap(args: InferenceConfig, tokenizer: Tokenizer):
    if args.beatmap_path is None or args.beatmap_path == "":
        return

    beatmap_path = Path(args.beatmap_path)

    if not beatmap_path.is_file():
        raise FileNotFoundError(f"Beatmap file {beatmap_path} not found.")

    beatmap = Beatmap.from_path(beatmap_path)
    print(f"Using metadata from beatmap: {beatmap.display_name}")

    if args.audio_path == '':
        args.audio_path = beatmap_path.parent / beatmap.audio_filename
    if args.output_path == '':
        args.output_path = beatmap_path.parent

    generation_config = generation_config_from_beatmap(beatmap, tokenizer)

    if args.gamemode is None:
        args.gamemode = generation_config.gamemode
        print(f"Using game mode {args.gamemode}")
    if args.beatmap_id is None and generation_config.beatmap_id:
        args.beatmap_id = generation_config.beatmap_id
        print(f"Using beatmap ID {args.beatmap_id}")
    if args.difficulty is None and generation_config.difficulty != -1 and len(beatmap.hit_objects(stacking=False)) > 0:
        args.difficulty = generation_config.difficulty
        print(f"Using difficulty {args.difficulty}")
    if args.mapper_id is None and beatmap.beatmap_id in tokenizer.beatmap_mapper:
        args.mapper_id = generation_config.mapper_id
        print(f"Using mapper ID {args.mapper_id}")
    if args.descriptors is None and beatmap.beatmap_id in tokenizer.beatmap_descriptors:
        args.descriptors = generation_config.descriptors
        print(f"Using descriptors {args.descriptors}")
    if args.circle_size is None:
        args.circle_size = generation_config.circle_size
        print(f"Using circle size {args.circle_size}")
    if args.slider_multiplier is None:
        args.slider_multiplier = generation_config.slider_multiplier
        print(f"Using slider multiplier {args.slider_multiplier}")
    if args.hitsounded is None:
        args.hitsounded = generation_config.hitsounded
        print(f"Using hitsounded {args.hitsounded}")
    if args.keycount is None and args.gamemode == 3:
        args.keycount = int(generation_config.keycount)
        print(f"Using keycount {args.keycount}")
    if args.hold_note_ratio is None and args.gamemode == 3:
        args.hold_note_ratio = generation_config.hold_note_ratio
        print(f"Using hold note ratio {args.hold_note_ratio}")
    if args.scroll_speed_ratio is None and args.gamemode == 3:
        args.scroll_speed_ratio = generation_config.scroll_speed_ratio
        print(f"Using scroll speed ratio {args.scroll_speed_ratio}")

    beatmap_config = beatmap_config_from_beatmap(beatmap)

    args.title = beatmap_config.title
    args.artist = beatmap_config.artist
    args.bpm = beatmap_config.bpm
    args.offset = beatmap_config.offset
    args.background = beatmap.background
    args.preview_time = beatmap_config.preview_time


def get_tags_dict(args: DictConfig | InferenceConfig):
    return dict(
        lookback=args.lookback,
        lookahead=args.lookahead,
        beatmap_id=args.beatmap_id,
        difficulty=args.difficulty,
        mapper_id=args.mapper_id,
        year=args.year,
        hitsounded=args.hitsounded,
        hold_note_ratio=args.hold_note_ratio,
        scroll_speed_ratio=args.scroll_speed_ratio,
        descriptors=f"\"[{','.join(args.descriptors)}]\"" if args.descriptors else None,
        negative_descriptors=f"\"[{','.join(args.negative_descriptors)}]\"" if args.negative_descriptors else None,
        timing_leniency=args.timing_leniency,
        seed=args.seed,
        add_to_beatmap=args.add_to_beatmap,
        start_time=args.start_time,
        end_time=args.end_time,
        in_context=f"[{','.join(ctx.value.upper() if isinstance(ctx, ContextType) else ctx for ctx in args.in_context)}]",
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        timing_temperature=args.timing_temperature,
        mania_column_temperature=args.mania_column_temperature,
        taiko_hit_temperature=args.taiko_hit_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        parallel=args.parallel,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        super_timing=args.super_timing,
        timer_num_beams=args.timer_num_beams,
        timer_bpm_threshold=args.timer_bpm_threshold,
        timer_cfg_scale=args.timer_cfg_scale,
        timer_iterations=args.timer_iterations,
        generate_positions=args.generate_positions,
        diff_cfg_scale=args.diff_cfg_scale,
        max_seq_len=args.max_seq_len,
        overlap_buffer=args.overlap_buffer,
    )


def get_config(args: InferenceConfig):
    # Create tags that describes args
    tags = get_tags_dict(args)
    # Filter to all non-default values
    defaults = get_tags_dict(OmegaConf.load("configs/inference.yaml"))
    tags = {k: v for k, v in tags.items() if v != defaults[k]}
    # To string separated by spaces
    tags = " ".join(f"{k}={v}" for k, v in tags.items())

    # Set defaults for generation config that does not allow an unknown value
    return GenerationConfig(
        gamemode=args.gamemode if args.gamemode is not None else 0,
        beatmap_id=args.beatmap_id,
        difficulty=args.difficulty,
        mapper_id=args.mapper_id,
        year=args.year,
        hitsounded=args.hitsounded if args.hitsounded is not None else True,
        slider_multiplier=args.slider_multiplier or 1.4,
        circle_size=args.circle_size,
        keycount=args.keycount if args.keycount is not None else 4,
        hold_note_ratio=args.hold_note_ratio,
        scroll_speed_ratio=args.scroll_speed_ratio,
        descriptors=args.descriptors,
        negative_descriptors=args.negative_descriptors,
    ), BeatmapConfig(
        title=args.title,
        artist=args.artist,
        title_unicode=args.title,
        artist_unicode=args.artist,
        audio_filename=Path(args.audio_path).name,
        circle_size=(args.keycount if args.gamemode == 3 else args.circle_size) or 4,
        slider_multiplier=args.slider_multiplier or 1.4,
        creator=args.creator,
        version=args.version,
        tags=tags,
        background_line=background_line(args.background),
        preview_time=args.preview_time,
        bpm=args.bpm,
        offset=args.offset,
        mode=args.gamemode,
    )


def generate(
        args: InferenceConfig,
        *,
        audio_path: str = None,
        beatmap_path: str = None,
        generation_config: GenerationConfig,
        beatmap_config: BeatmapConfig,
        model,
        tokenizer,
        diff_model=None,
        diff_tokenizer=None,
        refine_model=None,
        verbose=True,
):
    audio_path = args.audio_path if audio_path is None else audio_path
    beatmap_path = args.beatmap_path if beatmap_path is None else beatmap_path

    preprocessor = Preprocessor(args, parallel=args.parallel)
    processor = Processor(args, model, tokenizer)
    postprocessor = Postprocessor(args)

    audio = preprocessor.load(audio_path)
    sequences = preprocessor.segment(audio)
    extra_in_context = {}
    output_type = args.output_type.copy()

    # Auto generate timing if not provided in in_context and required for the model and this output_type
    timing_events, timing_times, timing = None, None, None
    if args.super_timing and ContextType.NONE in args.in_context:
        super_timing_generator = SuperTimingGenerator(args, model, tokenizer)
        timing_events, timing_times = super_timing_generator.generate(audio, generation_config, verbose=verbose)
        timing = postprocessor.generate_timing(timing_events)
        extra_in_context[ContextType.TIMING] = timing
        if ContextType.TIMING in output_type:
            output_type.remove(ContextType.TIMING)
    elif (ContextType.NONE in args.in_context and ContextType.MAP in output_type and
          not any("none" in ctx["in"] and ctx["out"] == "map" for ctx in args.osut5.data.context_types)):
        # Generate timing and convert in_context to timing context
        timing_events, timing_times = processor.generate(
            sequences=sequences,
            generation_config=generation_config,
            in_context=[ContextType.NONE],
            out_context=[ContextType.TIMING],
            verbose=verbose,
        )[0]
        timing_events, timing_times = events_of_type(timing_events, timing_times, TIMING_TYPES)
        timing = postprocessor.generate_timing(timing_events)
        extra_in_context[ContextType.TIMING] = timing
        if ContextType.TIMING in output_type:
            output_type.remove(ContextType.TIMING)
    elif ContextType.TIMING in args.in_context or (
            args.osut5.data.add_timing and any(t in args.in_context for t in [ContextType.GD, ContextType.NO_HS])):
        # Exact timing is provided in the other beatmap, so we don't need to generate it
        timing = [tp for tp in Beatmap.from_path(Path(beatmap_path)).timing_points if tp.parent is None]

    # Generate beatmap
    if len(output_type) > 0:
        result = processor.generate(
            sequences=sequences,
            generation_config=generation_config,
            in_context=args.in_context,
            out_context=output_type,
            beatmap_path=beatmap_path,
            extra_in_context=extra_in_context,
            verbose=verbose,
        )

        events, _ = reduce(merge_events, result)

        if timing is None and (ContextType.TIMING in args.output_type or args.osut5.data.add_timing):
            timing = postprocessor.generate_timing(events)

        # Resnap timing events
        if timing is not None:
            events = postprocessor.resnap_events(events, timing)
    else:
        events = timing_events

    # Generate positions with diffusion
    if args.generate_positions and args.gamemode in [0, 2] and ContextType.MAP in output_type:
        diffusion_pipeline = DiffisionPipeline(args, diff_model, diff_tokenizer, refine_model)
        events = diffusion_pipeline.generate(
            events=events,
            generation_config=generation_config,
            timing=timing,
            verbose=verbose,
        )

    result = postprocessor.generate(
        events=events,
        beatmap_config=beatmap_config,
        timing=timing,
    )

    result_path = None
    osz_path = None
    if args.add_to_beatmap:
        result_path = postprocessor.add_to_beatmap(result, beatmap_path)
        if verbose:
            print(f"Added generated content to {result_path}")
    elif args.output_path is not None and args.output_path != "":
        result_path = postprocessor.write_result(result, args.output_path)
        if verbose:
            print(f"Generated beatmap saved to {result_path}")

    if args.export_osz:
        osz_path = postprocessor.export_osz(result_path, audio_path, args.output_path)
        if verbose:
            print(f"Generated .osz saved to {osz_path}")

    return result, result_path, osz_path


def load_model(
        ckpt_path: str,
        t5_args: TrainConfig,
        device,
):
    if not os.path.exists(ckpt_path) and ckpt_path != "":
        model = Mapperatorinator.from_pretrained(ckpt_path)
        model.generation_config.disable_compile = True
        tokenizer = Tokenizer.from_pretrained(ckpt_path)
    else:
        ckpt_path = Path(ckpt_path)
        model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device, weights_only=True)
        tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=routed_pickle, weights_only=False)

        tokenizer = Tokenizer()
        tokenizer.load_state_dict(tokenizer_state)

        model = get_model(t5_args, tokenizer)
        model.load_state_dict(model_state)

    model.eval()
    model.to(torch.bfloat16).to(device)

    return model, tokenizer


def load_diff_model(
        ckpt_path,
        diff_args: DiffusionTrainConfig,
        device,
):
    if not os.path.exists(ckpt_path) and ckpt_path != "":
        tokenizer_file = cached_file(ckpt_path, "tokenizer.pkl")
        model_file = cached_file(ckpt_path, "model_ema.pkl")
    else:
        ckpt_path = Path(ckpt_path)
        tokenizer_file = ckpt_path / "tokenizer.pkl"
        model_file = ckpt_path / "model_ema.pkl"

    tokenizer_state = torch.load(tokenizer_file, pickle_module=routed_pickle, weights_only=False)
    tokenizer = osu_diffusion.utils.tokenizer.Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    ema_state = torch.load(model_file, pickle_module=routed_pickle, weights_only=False, map_location=device)
    model = DiT_models[diff_args.model.model](
        context_size=diff_args.model.context_size,
        class_size=tokenizer.num_tokens,
    ).to(device)
    model.load_state_dict(ema_state)
    model.eval()  # important!
    return model, tokenizer


@hydra.main(config_path="configs", config_name="inference_v30", version_base="1.1")
def main(args: InferenceConfig):
    prepare_args(args)

    model, tokenizer = load_model(args.model_path, args.osut5, args.device)

    diff_model, diff_tokenizer, refine_model = None, None, None
    if args.generate_positions:
        diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion, args.device)

        if os.path.exists(args.diff_refine_ckpt):
            refine_model = load_diff_model(args.diff_refine_ckpt, args.diffusion, args.device)[0]

        if args.compile:
            diff_model.forward = torch.compile(diff_model.forward, mode="reduce-overhead", fullgraph=True)

    get_args_from_beatmap(args, tokenizer)
    generation_config, beatmap_config = get_config(args)

    return generate(
        args,
        generation_config=generation_config,
        beatmap_path=args.beatmap_path,
        beatmap_config=beatmap_config,
        model=model,
        tokenizer=tokenizer,
        diff_model=diff_model,
        diff_tokenizer=diff_tokenizer,
        refine_model=refine_model,
    )


if __name__ == "__main__":
    main()
