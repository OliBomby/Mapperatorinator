from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from string import Template

from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich.console import Console

import hydra
from slider import Beatmap, Spinner

from config import MaiModConfig
from inference import prepare_args, get_args_from_beatmap, get_config, load_model_with_server
from osuT5.osuT5.dataset.data_utils import get_groups, Group
from osuT5.osuT5.event import EventType, Event, ContextType
from osuT5.osuT5.inference import Preprocessor, Processor, GenerationConfig
from osuT5.osuT5.inference.server import InferenceClient
from osuT5.osuT5.model import Mapperatorinator

# These event types are designed for V30 tokenization
mod_explanations = {
    # Real, Expected
    (EventType.DISTANCE, EventType.DISTANCE): ("Compose", "Expected distance $expected_value to the previous $previous_group instead of $real_value."),
    (EventType.POS_X, EventType.POS_X): ("Compose", "Expected position $expected_value instead of $real_value."),
    (EventType.POS_Y, EventType.POS_Y): ("Compose", "Expected position $expected_value instead of $real_value."),
    (EventType.POS, EventType.POS): ("Compose", "Expected position $expected_value instead of $real_value."),
    (EventType.MANIA_COLUMN, EventType.MANIA_COLUMN): ("Compose", "Expected column $expected_value instead of $real_value."),
    (EventType.HITSOUND, EventType.HITSOUND): ("Hit Sounds", "Expected hitsound $expected_value instead of $real_value."),
    (EventType.VOLUME, EventType.VOLUME): ("Hit Sounds", "Expected volume $expected_value instead of $real_value."),
    (EventType.HITSOUND, EventType.NEW_COMBO): ("New Combos", "Expected new combo."),
    (EventType.NEW_COMBO, EventType.HITSOUND): ("New Combos", "Unexpected new combo."),
    (EventType.HITSOUND, EventType.LAST_ANCHOR): ("Rhythm", "Expected end of slider repeats."),  # Types last
    (EventType.HITSOUND, EventType.SLIDER_END): ("Rhythm", "Expected end of slider repeats."),  # Types first
    (EventType.SNAPPING, EventType.BEAT): ("Timing", "Hit object likely not snapped to a beat."),
    (EventType.SNAPPING, EventType.MEASURE): ("Timing", "Hit object likely not snapped to a beat."),
    (EventType.SNAPPING, EventType.TIMING_POINT): ("Timing", "Hit object likely not snapped to a beat."),
    (EventType.TIME_SHIFT, EventType.DISTANCE): ("Sliders", "Expected additional anchors."),
    (EventType.DISTANCE, EventType.TIME_SHIFT): ("Sliders", "Expected last anchor."),
    (EventType.BEAT, EventType.SNAPPING): ("Timing", "Unexpected beat."),
    (EventType.BEAT, EventType.MEASURE): ("Timing", "Expected new measure."),
    (EventType.BEAT, EventType.TIMING_POINT): ("Timing", "Expected new timing point."),
    (EventType.MEASURE, EventType.SNAPPING): ("Timing", "Unexpected new measure."),
    (EventType.MEASURE, EventType.BEAT): ("Timing", "Unexpected new measure."),
    (EventType.MEASURE, EventType.TIMING_POINT): ("Timing", "Expected new timing point."),
    (EventType.TIMING_POINT, EventType.SNAPPING): ("Timing", "Unexpected new timing point."),
    (EventType.TIMING_POINT, EventType.BEAT): ("Timing", "Unexpected new timing point."),
    (EventType.TIMING_POINT, EventType.MEASURE): ("Timing", "Unexpected new timing point."),
}


@dataclass
class Suggestion:
    context_type: ContextType
    index: int
    time: float
    group: Group
    group_str: str
    previous_group_str: str
    next_group: Group | None
    next_beat_group: Group | None
    event: Event
    event_str: str
    expected_event: Event
    expected_event_str: str
    surprisal: float
    combo_index: int | None = None
    timestamp_time: float | None = None


def type_to_str(event_type: EventType) -> str:
    return event_type.value.replace("_", " ").title()


def ai_mod(
        args: MaiModConfig,
        *,
        audio_path: str = None,
        beatmap_path: str = None,
        generation_config: GenerationConfig,
        model: Mapperatorinator | InferenceClient,
        tokenizer,
        verbose=True,
):
    i_args = args.inference
    audio_path = i_args.audio_path if audio_path is None else audio_path
    beatmap_path = i_args.beatmap_path if beatmap_path is None else beatmap_path

    # Do some validation
    if not Path(audio_path).exists() or not Path(audio_path).is_file():
        raise FileNotFoundError(f"Provided audio file path does not exist: {audio_path}")
    if beatmap_path:
        beatmap_path_obj = Path(beatmap_path)
        if not beatmap_path_obj.exists() or not beatmap_path_obj.is_file():
            raise FileNotFoundError(f"Provided beatmap file path does not exist: {beatmap_path}")
        # Validate beatmap file type
        if beatmap_path_obj.suffix.lower() != '.osu':
            raise ValueError(f"Beatmap file must have .osu extension: {beatmap_path}")

    preprocessor = Preprocessor(i_args, parallel=False)
    processor = Processor(i_args, model, tokenizer)

    audio = preprocessor.load(audio_path)
    sequences = preprocessor.segment(audio)

    # Generate logits
    result = processor.ai_mod(
        sequences=sequences,
        generation_config=generation_config,
        beatmap_path=beatmap_path,
        verbose=verbose,
    )

    position_types = [EventType.DISTANCE, EventType.POS_X, EventType.POS_Y, EventType.POS]
    anchor_types = [EventType.RED_ANCHOR, EventType.BEZIER_ANCHOR, EventType.CATMULL_ANCHOR, EventType.PERFECT_ANCHOR]
    hs_types = [EventType.HITSOUND, EventType.VOLUME]
    timing_types = [EventType.BEAT, EventType.MEASURE, EventType.TIMING_POINT]
    hitobject_types = [EventType.CIRCLE, EventType.SPINNER, EventType.SPINNER_END, EventType.SLIDER_HEAD, EventType.BEZIER_ANCHOR, EventType.PERFECT_ANCHOR, EventType.CATMULL_ANCHOR, EventType.RED_ANCHOR, EventType.LAST_ANCHOR, EventType.SLIDER_END, EventType.HOLD_NOTE, EventType.HOLD_NOTE_END, EventType.DRUMROLL, EventType.DRUMROLL_END, EventType.DENDEN, EventType.DENDEN_END]

    # Print for every context and every event type, the top 10 events with the highest surprisal
    # Also skip anything below 1 relative suprisal
    suggestions: list[Suggestion] = []
    for context in result:
        groups, group_indices = get_groups(context['events'], event_times=context['event_times'], types_first=i_args.train.data.types_first)
        # Group indices map each group index to a list of indices of the events in the original list
        # We need the reverse mapping to get the groups for each event
        event_groups: list[int] = [0] * len(context['events'])
        for group_index, indices in enumerate(group_indices):
            for index in indices:
                event_groups[index] = group_index

        context_suggestions = [
            Suggestion(context['context_type'], *z) for z in zip(
                range(len(context['events'])),
                context['event_times'],
                [groups[event_groups[i]] for i in range(len(context['events']))],
                ["None"] * len(context['events']),
                ["None"] * len(context['events']),
                [groups[event_groups[i] + 1] if event_groups[i] + 1 < len(groups) else None for i in range(len(context['events']))],
                [None] * len(context['events']),
                context['events'],
                context['events_str'],
                context['expected_events'],
                context['expected_events_str'],
                context['surprisals'],
                [None] * len(context['events']),
                [None] * len(context['events']),
            )
        ]

        def get_group_str(group_index: int, s: Suggestion) -> str:
            if group_index < 0 or group_index >= len(groups):
                return "None"
            group = groups[group_index]
            if group.event_type == EventType.LAST_ANCHOR and group_index == event_groups[s.index] and s.event.type in hs_types:
                # This group is for a hitsound event on a slider end group, which contains hitsound events for each repeat
                # Find the repeat index this hitsound event corresponds to
                repeat_index = 0
                for j in range(s.index - 1, -1, -1):
                    if context['events'][j].type == EventType.TIME_SHIFT:
                        break
                    if context['events'][j].type == s.event.type:
                        repeat_index += 1

                if repeat_index == 0:
                    return "Slider Body"
                else:
                    return f"Slider Repeat #{repeat_index}"
            elif group.event_type in anchor_types:
                # Count the number of anchor groups in between this group and the slider head group
                anchor_index = 2
                for j in range(group_index - 1, -1, -1):
                    if groups[j].event_type == EventType.SLIDER_HEAD:
                        break
                    if groups[j].event_type in anchor_types:
                        anchor_index += 1
                return f"{type_to_str(group.event_type)} #{anchor_index}"
            else:
                return type_to_str(group.event_type)

        def get_next_beat_group(s: Suggestion) -> Group | None:
            # Find the next group that is a beat, measure, or timing point
            for i in range(event_groups[s.index] + 1, len(groups)):
                if groups[i].event_type in timing_types:
                    return groups[i]
            return None

        # If the group is an anchor, we want to print the anchor index in the slider
        for s in context_suggestions:
            group_index = event_groups[s.index]
            s.group_str = get_group_str(group_index, s)
            s.next_beat_group = get_next_beat_group(s)

            # Find the previous group with positions
            for i in range(group_index - 1, -1, -1):
                if groups[i].x is not None:
                    s.previous_group_str = get_group_str(i, s)
                    break

        suggestions.extend(context_suggestions)

    suggestions.sort(key=lambda x: x.time)

    # Filter suggestions
    suggestions = [
        s for s in suggestions
        if (s.surprisal >= 20.0 and
            not (s.group.event_type == EventType.SLIDER_END and s.event.type in position_types) and
            not (s.event.type == EventType.TIME_SHIFT and s.expected_event.type == EventType.TIME_SHIFT and abs(s.expected_event.value - s.event.value) <= 10) and
            not (s.event.type == EventType.SNAPPING and s.expected_event.type in timing_types and s.next_group and abs(s.time - s.next_group.time) < 2) and
            not (s.event.type in timing_types and s.expected_event.type == EventType.SNAPPING and s.next_group and abs(s.time - s.next_group.time) < 2))
    ]

    # Add hit object combo index to each suggestion with a hit object related group
    beatmap = Beatmap.from_path(beatmap_path)
    hitobjects = beatmap.hit_objects(stacking=False)
    for s in suggestions:
        if s.group.event_type not in hitobject_types:
            continue
        # Find the hit object that corresponds to this group and its combo index
        combo_index = 0
        for i, hitobject in enumerate(hitobjects):
            if hitobject.time.total_seconds() * 1000 - 1 > s.time and i > 0:
                s.combo_index = combo_index
                s.timestamp_time = int(hitobjects[i - 1].time.total_seconds() * 1000 + 1e-5)
                break
            combo_index += 1
            if hitobject.new_combo or isinstance(hitobject, Spinner) or (i > 0 and isinstance(hitobjects[i - 1], Spinner)) or (i > 0 and hitobject.time - hitobjects[i - 1].time > timedelta(seconds=10)):
                combo_index = 1
        if combo_index > 0 and s.combo_index is None:
            s.combo_index = combo_index
            s.timestamp_time = int(hitobjects[-1].time.total_seconds() * 1000 + 1e-5)

    def timestamp_text(s: Suggestion) -> str:
        t = s.time
        t2 = s.timestamp_time if s.timestamp_time is not None else t
        timestamp = f"{t // 60000:02}:{(t // 1000) % 60:02}:{t % 1000:03}"
        url = f"osu://edit/{t2 // 60000:02}:{(t2 // 1000) % 60:02}:{t2 % 1000:03}"
        if s.combo_index is not None:
            url += f"%20({s.combo_index})"
            # timestamp += f" ({s.combo_index})"
        return f"[link={url}][green]{timestamp}[/green][/link]"

    def surprisal_text(surprisal: float) -> str:
        surprisal /= 10.0  # Normalize surprisal to a more readable scale
        if surprisal >= 10000:
            return f"[bold red]({surprisal:.0f})[/bold red]"
        elif surprisal >= 1000:
            return f" [bold red]({surprisal:.0f})[/bold red]"
        elif surprisal >= 100:
            return f"  [bold red]({surprisal:.0f})[/bold red]"
        elif surprisal >= 10:
            return f"   [bold yellow]({surprisal:.0f})[/bold yellow]"
        elif surprisal >= 1:
            return f"    [bold]({surprisal:.0f})[/bold]"
        else:
            return f"    ({surprisal:.0f})"

    suggestions_by_category = {}

    for s in suggestions:
        if i_args.train.data.add_timing and s.event.type == EventType.TIME_SHIFT and s.expected_event.type == EventType.TIME_SHIFT and s.group.event_type not in timing_types and s.next_beat_group and abs(s.expected_event.value - s.next_beat_group.time) <= 10:
            # The model predicted the time of the next beat, so it expects no hit object here
            category, explanation_template = ("Rhythm", "Unexpected hit object.")
        elif s.event.type == EventType.LAST_ANCHOR and s.expected_event.type in anchor_types:
            category, explanation_template = ("Sliders", "Expected additional anchors.")
        elif s.event.type in anchor_types and s.expected_event.type == EventType.LAST_ANCHOR:
            category, explanation_template = ("Sliders", "Expected last anchor.")
        elif s.event.type in anchor_types and s.expected_event.type in anchor_types:
            category, explanation_template = ("Sliders", "Expected a $expected_type instead of a $real_type.")
        elif s.event.type in hitobject_types and s.expected_event.type in hitobject_types:
            category, explanation_template = ("Rhythm", "Expected a $expected_type instead of a $real_type.")
        elif s.event.type in [EventType.TIME_SHIFT, EventType.SNAPPING] and s.expected_event.type == s.event.type:
            if s.event.type == EventType.TIME_SHIFT:
                explanation_template = "Expected object at $expected_value instead of $real_value."
            else:
                explanation_template = "Expected snapping $expected_value instead of $real_value."
            if s.group.event_type in hitobject_types:
                category = "Rhythm"
            elif s.group.event_type == EventType.SCROLL_SPEED_CHANGE:
                category = "Scroll Speeds"
            elif s.group.event_type == EventType.KIAI:
                category = "Kiai"
            else:
                category = "Timing"
        elif s.event.type == EventType.SCROLL_SPEED and s.expected_event.type == EventType.SCROLL_SPEED:
            if beatmap.mode == 0:
                # In osu!standard, scroll speed is called 'Slider Velocity'
                category, explanation_template = ("Sliders", "Expected slider velocity $expected_value instead of $real_value.")
            else:
                # In any other game mode, scroll speed is called 'Scroll Speed'
                category, explanation_template = ("Scroll Speeds", "Expected scroll speed $expected_value instead of $real_value.")
        elif s.expected_event.type == EventType.CONTROL:
            if s.event.type == EventType.KIAI:
                if s.event.value == 1:
                    category, explanation_template = ("Kiai", "Unexpected kiai section start.")
                else:
                    category, explanation_template = ("Kiai", "Unexpected kiai section end.")
            else:
                category, explanation_template = ("Timing", "Expected end of beatmap.")
        else:
            category, explanation_template = mod_explanations.get((s.event.type, s.expected_event.type), ("Misc", "Expected $expected_type $expected_value instead of $real_type $real_value."))

        explanation_template = Template(explanation_template)
        explanation = explanation_template.safe_substitute({
            'expected_value': s.expected_event_str,
            'real_value': s.event_str,
            'expected_type': type_to_str(s.expected_event.type),
            'real_type': type_to_str(s.event.type),
            'group': s.group_str,
            'previous_group': s.previous_group_str,
        })

        if category not in suggestions_by_category:
            suggestions_by_category[category] = []
        suggestions_by_category[category].append(f"{surprisal_text(s.surprisal)} {timestamp_text(s)} ({s.group_str}) - {explanation}")

    # Print the suggestions by category
    console = Console(width=900)
    p = print if args.raw_output else console.print

    categories = sorted(suggestions_by_category.keys())
    p("The first value between parentheses represents the importance of the suggestion. Values above [red]100[/red] are likely issues, whereas values below 10 are likely subjective")
    p(f"Found {len(suggestions)} suggestions:")
    for category in categories:
        print(f"\n{category}:")
        for item in suggestions_by_category[category][:10]:
            p(f" {item}")


@hydra.main(config_path="configs", config_name="mai_mod", version_base="1.1")
def main(args: MaiModConfig):
    # Select inference config based on the beatmap gamemode
    if args.beatmap_path:
        beatmap_path = Path(args.beatmap_path)
        if not beatmap_path.exists() or not beatmap_path.is_file():
            raise FileNotFoundError(f"Provided beatmap file path does not exist: {args.beatmap_path}")
        if beatmap_path.suffix.lower() != '.osu':
            raise ValueError(f"Beatmap file must have .osu extension: {args.beatmap_path}")

        # Load the beatmap to determine the gamemode
        beatmap = Beatmap.from_path(beatmap_path)

        if beatmap.mode in args.inference.train.data.gamemodes:
            # We can use the current inference config
            pass
        else:
            # Fallback to V31
            original_cli_overrides = HydraConfig.get().overrides.task
            merged_overrides = ["inference=v31"]
            merged_overrides.extend(original_cli_overrides)
            print(f"\nOriginal Command-Line Overrides: {original_cli_overrides}")
            GlobalHydra.instance().clear()
            with hydra.initialize(version_base="1.1", config_path="configs"):
                conf = hydra.compose(config_name="mai_mod", overrides=merged_overrides)
                args.inference = OmegaConf.to_object(conf).inference

    i_args = args.inference
    i_args.beatmap_path = args.beatmap_path
    i_args.audio_path = args.audio_path
    i_args.precision = args.precision

    prepare_args(i_args)

    model, tokenizer = load_model_with_server(
        i_args.model_path,
        i_args.train,
        i_args.device,
        max_batch_size=i_args.max_batch_size,
        use_server=False,
    )

    get_args_from_beatmap(i_args, tokenizer)
    generation_config, beatmap_config = get_config(i_args)

    return ai_mod(
        args,
        generation_config=generation_config,
        beatmap_path=args.beatmap_path,
        model=model,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
