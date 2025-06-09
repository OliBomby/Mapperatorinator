from pathlib import Path
from typing import Optional

import numpy as np
from pydub import AudioSegment

import numpy.typing as npt
from ..tokenizer import Event, EventType

MILISECONDS_PER_SECOND = 1000




def load_audio_file(file: Path, sample_rate: int, speed: float = 1.0) -> npt.NDArray:
    print(f"Loading audio file {file}")
    try:
        file_path = Path(file).resolve().as_posix()
        audio = AudioSegment.from_file(str(file_path), format="mp3")
        audio.frame_rate = int(audio.frame_rate * speed)
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples *= 1.0 / np.max(np.abs(samples))
        return samples
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())  # Print the traceback for debugging
        return None

def update_event_times(events: list[Event], event_times: list[float], end_time: Optional[float] = None):
    non_timed_events = [
        EventType.BEZIER_ANCHOR,
        EventType.PERFECT_ANCHOR,
        EventType.CATMULL_ANCHOR,
        EventType.RED_ANCHOR,
    ]
    timed_events = [
        EventType.CIRCLE,
        EventType.SPINNER,
        EventType.SPINNER_END,
        EventType.SLIDER_HEAD,
        EventType.LAST_ANCHOR,
        EventType.SLIDER_END,
        EventType.BEAT,
        EventType.MEASURE,
    ]

    start_index = len(event_times)
    end_index = len(events)
    ct = 0 if len(event_times) == 0 else event_times[-1]
    for i in range(start_index, end_index):
        event = events[i]
        if event.type == EventType.TIME_SHIFT:
            ct = event.value
        event_times.append(ct)

    # Interpolate time for control point events
    # T-D-Start-D-CP-D-CP-T-D-LCP-T-D-End
    # 1-1-1-----1-1--1-1--7-7--7--9-9-9--
    # 1-1-1-----3-3--5-5--7-7--7--9-9-9--
    ct = end_time if end_time is not None else event_times[-1]
    interpolate = False
    for i in range(end_index - 1, start_index - 1, -1):
        event = events[i]

        if event.type in timed_events:
            interpolate = False

        if event.type in non_timed_events:
            interpolate = True

        if not interpolate:
            ct = event_times[i]
            continue

        if event.type not in non_timed_events:
            event_times[i] = ct
            continue

        # Find the time of the first timed event and the number of control points between
        j = i
        count = 0
        t = ct
        while j >= 0:
            event2 = events[j]
            if event2.type == EventType.TIME_SHIFT:
                t = event_times[j]
                break
            if event2.type in non_timed_events:
                count += 1
            j -= 1
        if i < 0:
            t = 0

        # Interpolate the time
        ct = (ct - t) / (count + 1) * count + t
        event_times[i] = ct


def merge_events(events1: list[Event], events2: list[Event]) -> list[Event]:
    """Merge two lists of events in a time sorted manner. Assumes both lists are sorted by time.

    Args:
        events1: List of events.
        events2: List of events.

    Returns:
        merged_events: Merged list of events.
    """
    merged_events = []
    i = 0
    j = 0
    t1 = 0
    t2 = 0

    while i < len(events1) and j < len(events2):
        if events1[i].type == EventType.TIME_SHIFT:
            t1 = events1[i].value
        if events2[j].type == EventType.TIME_SHIFT:
            t2 = events2[j].value

        if t1 <= t2:
            merged_events.append(events1[i])
            i += 1
        else:
            merged_events.append(events2[j])
            j += 1

    merged_events.extend(events1[i:])
    merged_events.extend(events2[j:])
    return merged_events


def remove_events_of_type(events: list[Event], event_types: list[EventType]) -> list[Event]:
    """Remove all events of a specific type from a list of events.

    Args:
        events: List of events.
        event_types: Types of event to remove.

    Returns:
        filtered_events: Filtered list of events.
    """
    return [event for event in events if event.type not in event_types]


def speed_events(events: list[Event], speed: float) -> list[Event]:
    """Change the speed of a list of events.

    Args:
        events: List of events.
        speed: Speed multiplier.

    Returns:
        sped_events: Sped up list of events.
    """
    sped_events = []
    for event in events:
        if event.type == EventType.TIME_SHIFT:
            event.value = int(event.value / speed)
        sped_events.append(event)
    return sped_events
