import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment

import numpy.typing as npt
from slider import Beatmap, HoldNote, TimingPoint

from ..tokenizer import Event, EventType

MILISECONDS_PER_SECOND = 1000
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


def load_audio_file(file: str, sample_rate: int, speed: float = 1.0) -> npt.NDArray:
    """Load an audio file as a numpy time-series array

    The signals are resampled, converted to mono channel, and normalized.

    Args:
        file: Path to audio file.
        sample_rate: Sample rate to resample the audio.
        speed: Speed multiplier for the audio.

    Returns:
        samples: Audio time series.
    """
    file = Path(file)
    audio = AudioSegment.from_file(file)
    audio.frame_rate = int(audio.frame_rate * speed)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples *= 1.0 / np.max(np.abs(samples))
    return samples


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


def get_hold_note_ratio(beatmap: Beatmap) -> float:
    notes = beatmap.hit_objects(stacking=False)
    hold_note_count = 0
    for note in notes:
        if isinstance(note, HoldNote):
            hold_note_count += 1
    return hold_note_count / len(notes)


def get_scroll_speed_ratio(beatmap: Beatmap) -> float:
    # Number of scroll speed changes divided by number of distinct hit object times
    notes = beatmap.hit_objects(stacking=False)
    last_time = -1
    num_note_times = 0
    for note in notes:
        if note.time != last_time:
            num_note_times += 1
            last_time = note.time
    last_scroll_speed = -1
    num_scroll_speed_changes = 0
    for timing_point in beatmap.timing_points:
        if timing_point.parent is None:
            last_scroll_speed = 1
        else:
            scroll_speed = -100 / timing_point.ms_per_beat
            if scroll_speed != last_scroll_speed and last_scroll_speed != -1:
                num_scroll_speed_changes += 1
            last_scroll_speed = scroll_speed
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
