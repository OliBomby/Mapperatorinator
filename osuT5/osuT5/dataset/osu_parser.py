from __future__ import annotations

from datetime import timedelta
from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
from slider import Beatmap, Circle, Slider, Spinner, HoldNote, TimingPoint
from slider.curve import Linear, Catmull, Perfect, MultiBezier

from ..tokenizer import Tokenizer
from ..event import Event, EventType
from .data_utils import merge_events, speed_events, get_median_mpb_beatmap
from ..config import TrainConfig


class OsuParser:
    def __init__(self, args: TrainConfig, tokenizer: Tokenizer) -> None:
        self.types_first = args.data.types_first
        self.add_timing = args.data.add_timing
        self.add_snapping = args.data.add_snapping
        self.add_timing_points = args.data.add_timing_points
        self.add_hitsounds = args.data.add_hitsounds
        self.add_distances = args.data.add_distances
        self.add_positions = args.data.add_positions
        self.add_kiai = args.data.add_kiai
        self.add_sv = args.data.add_sv
        self.add_mania_sv = args.data.add_mania_sv
        self.mania_bpm_normalized_scroll_speed = args.data.mania_bpm_normalized_scroll_speed
        self.position_precision = args.data.position_precision
        self.position_split_axes = args.data.position_split_axes
        self.x_min, self.x_max, self.y_min, self.y_max = args.data.position_range
        if self.add_distances:
            dist_range = tokenizer.event_range[EventType.DISTANCE]
            self.dist_min = dist_range.min_value
            self.dist_max = dist_range.max_value

    def parse(self, beatmap: Beatmap, speed: float = 1.0) -> tuple[list[Event], list[int]]:
        # noinspection PyUnresolvedReferences
        """Parse an .osu beatmap.

        Each hit object is parsed into a list of Event objects, in order of its
        appearance in the beatmap. In other words, in ascending order of time.

        Args:
            beatmap: Beatmap object parsed from an .osu file.
            speed: Speed multiplier for the beatmap.

        Returns:
            events: List of Event object lists.
            event_times: List of event times.

        Example::
            >>> beatmap = [
                "64,80,11000,1,0",
                "100,100,16000,2,0,B|200:200|250:200|250:200|300:150,2"
            ]
            >>> events, event_times = parse(beatmap)
            >>> print(events)
            [
                Event(EventType.TIME_SHIFT, 11000), Event(EventType.DISTANCE, 36), Event(EventType.CIRCLE),
                Event(EventType.TIME_SHIFT, 16000), Event(EventType.DISTANCE, 42), Event(EventType.SLIDER_HEAD),
                Event(EventType.DISTANCE, 141), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.DISTANCE, 50), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.DISTANCE, 10), Event(EventType.BEZIER_ANCHOR),
                Event(EventType.TIME_SHIFT, 18000), Event(EventType.DISTANCE, 64), Event(EventType.LAST _ANCHOR),
                Event(EventType.TIME_SHIFT, 20000), Event(EventType.DISTANCE, 11), Event(EventType.SLIDER_END)
            ]
        """
        hit_objects = beatmap.hit_objects(stacking=False)
        last_pos = np.array((256, 192))
        events = []
        event_times = []

        for hit_object in hit_objects:
            if isinstance(hit_object, Circle):
                last_pos = self._parse_circle(hit_object, events, event_times, last_pos, beatmap)
            elif isinstance(hit_object, Slider):
                if beatmap.mode == 1:
                    self._parse_drumroll(hit_object, events, event_times, beatmap)
                else:
                    last_pos = self._parse_slider(hit_object, events, event_times, last_pos, beatmap)
            elif isinstance(hit_object, Spinner):
                if beatmap.mode == 1:
                    self._parse_denden(hit_object, events, event_times, beatmap)
                else:
                    last_pos = self._parse_spinner(hit_object, events, event_times, beatmap)
            elif isinstance(hit_object, HoldNote):
                last_pos = self._parse_hold_note(hit_object, events, event_times, beatmap)

        # Sort events by time
        if len(events) > 0:
            events, event_times = zip(*sorted(zip(events, event_times), key=lambda x: x[1]))
        result = list(events), list(event_times)

        if self.add_mania_sv and beatmap.mode == 3:
            scroll_speed_events = self.parse_scroll_speeds(beatmap)
            result = merge_events(scroll_speed_events, result)

        if self.add_kiai:
            kiai_events = self.parse_kiai(beatmap)
            result = merge_events(kiai_events, result)

        if self.add_timing:
            timing_events = self.parse_timing(beatmap)
            result = merge_events(timing_events, result)

        if speed != 1.0:
            result = speed_events(result, speed)

        return result

    def parse_scroll_speeds(self, beatmap: Beatmap, speed: float = 1.0) -> tuple[list[Event], list[int]]:
        """Extract all BPM-normalized scroll speed changes from a beatmap."""
        normalized = self.mania_bpm_normalized_scroll_speed
        events = []
        event_times = []
        median_mpb = get_median_mpb_beatmap(beatmap)
        mpb = median_mpb
        last_normalized_scroll_speed = -1

        for i, tp in enumerate(beatmap.timing_points):
            if tp.parent is None:
                mpb = tp.ms_per_beat
                scroll_speed = 1
            else:
                scroll_speed = -100 / tp.ms_per_beat

            if i == len(beatmap.timing_points) - 1 or beatmap.timing_points[i + 1].offset > tp.offset:
                normalized_scroll_speed = scroll_speed * median_mpb / mpb if normalized else scroll_speed

                if normalized_scroll_speed != last_normalized_scroll_speed or last_normalized_scroll_speed == -1:
                    self._add_group(
                        EventType.SCROLL_SPEED_CHANGE,
                        tp.offset,
                        events,
                        event_times,
                        beatmap,
                        time_event=True,
                        scroll_speed=normalized_scroll_speed,
                    )
                last_normalized_scroll_speed = normalized_scroll_speed

        if speed != 1.0:
            events, event_times = speed_events((events, event_times), speed)

        return events, event_times

    def parse_kiai(self, beatmap: Beatmap, speed: float = 1.0) -> tuple[list[Event], list[int]]:
        """Extract all kiai information from a beatmap."""
        events = []
        event_times = []
        kiai = False

        for tp in beatmap.timing_points:
            if tp.kiai_mode == kiai:
                continue

            self._add_group(
                Event(EventType.KIAI, int(tp.kiai_mode)),
                tp.offset,
                events,
                event_times,
                beatmap,
                time_event=True,
            )
            kiai = tp.kiai_mode

        if speed != 1.0:
            events, event_times = speed_events((events, event_times), speed)

        return events, event_times

    def parse_timing(self, beatmap: Beatmap | list[TimingPoint], speed: float = 1.0, song_length: Optional[float] = None) -> tuple[list[Event], list[int]]:
        """Extract all timing information from a beatmap."""
        timing = beatmap.timing_points if isinstance(beatmap, Beatmap) else beatmap
        assert len(timing) > 0, "No timing points found in beatmap."

        events = []
        event_times = []
        if song_length is None and isinstance(beatmap, Beatmap):
            last_ho = beatmap.hit_objects(stacking=False)[-1]
            last_time = last_ho.end_time if hasattr(last_ho, "end_time") else last_ho.time
        elif song_length is not None:
            last_time = timedelta(milliseconds=song_length)
        else:
            last_time = timing[-1].offset + timedelta(minutes=10)

        # Get all timing points with BPM changes
        timing_points = [tp for tp in timing if tp.bpm]

        for i, tp in enumerate(timing_points):
            # Generate beat and measure events until the next timing point
            next_tp = timing_points[i + 1] if i + 1 < len(timing_points) else None
            next_time = next_tp.offset - timedelta(milliseconds=10) if next_tp else last_time
            time = tp.offset
            measure_counter = 0
            beat_delta = timedelta(milliseconds=tp.ms_per_beat)
            while time <= next_time:
                if self.add_timing_points and measure_counter == 0:
                    event_type = EventType.TIMING_POINT
                elif measure_counter % tp.meter == 0:
                    event_type = EventType.MEASURE
                else:
                    event_type = EventType.BEAT

                self._add_group(
                    event_type,
                    time,
                    events,
                    event_times,
                    beatmap,
                    time_event=True,
                    add_snap=False,
                )

                measure_counter += 1
                time += beat_delta

        if speed != 1.0:
            events, event_times = speed_events((events, event_times), speed)

        return events, event_times

    @staticmethod
    def uninherited_point_at(time: timedelta, beatmap: Beatmap):
        tp = beatmap.timing_point_at(time)
        return tp if tp.parent is None else tp.parent

    @staticmethod
    def hitsound_point_at(time: timedelta, beatmap: Beatmap):
        hs_query = time + timedelta(milliseconds=5)
        return beatmap.timing_point_at(hs_query)

    def scroll_speed_at(self, time: timedelta, beatmap: Beatmap) -> float:
        query = time
        tp = beatmap.timing_point_at(query)
        return self.tp_to_scroll_speed(tp)

    def tp_to_scroll_speed(self, tp: TimingPoint) -> float:
        if tp.parent is None or tp.ms_per_beat >= 0 or np.isnan(tp.ms_per_beat):
            return 1
        else:
            return np.clip(-100 / tp.ms_per_beat, 0.01, 10)

    def _add_time_event(self, time: timedelta, beatmap: Beatmap, events: list[Event], event_times: list[int], add_snap: bool = True) -> None:
        """Add a snapping event to the event list.

        Args:
            time: Time of the snapping event.
            beatmap: Beatmap object.
            events: List of events to add to.
            add_snap: Whether to add a snapping event.
        """
        time_ms = int(time.total_seconds() * 1000)
        events.append(Event(EventType.TIME_SHIFT, time_ms))
        event_times.append(time_ms)

        if not add_snap or not self.add_snapping:
            return

        tp = self.uninherited_point_at(time, beatmap)
        beats = (time - tp.offset).total_seconds() * 1000 / tp.ms_per_beat
        snapping = 0
        for i in range(1, 17):
            # If the difference between the time and the snapped time is less than 2 ms, that is the correct snapping
            if abs(beats - round(beats * i) / i) * tp.ms_per_beat < 2:
                snapping = i
                break

        events.append(Event(EventType.SNAPPING, snapping))
        event_times.append(time_ms)

    def _add_hitsound_event(self, time: timedelta, group_time: int, hitsound: int, addition: str, beatmap: Beatmap, events: list[Event], event_times: list[int]) -> None:
        if not self.add_hitsounds:
            return

        tp = self.hitsound_point_at(time, beatmap)
        tp_sample_set = tp.sample_type if tp.sample_type != 0 else 2  # Inherit to soft sample set
        addition_split = addition.split(":")
        sample_set = int(addition_split[0]) if addition_split[0] != "0" else tp_sample_set
        addition_set = int(addition_split[1]) if addition_split[1] != "0" else sample_set
        volume = int(addition_split[3]) if len(addition_split) > 3 and addition_split[3] != "0" else tp.volume

        sample_set = sample_set if 0 < sample_set < 4 else 1  # Overflow default to normal sample set
        addition_set = addition_set if 0 < addition_set < 4 else 1  # Overflow default to normal sample set
        hitsound = hitsound & 14  # Only take the bits for normal, whistle, and finish
        volume = np.clip(volume, 0, 100)

        hitsound_idx = hitsound // 2 + 8 * (sample_set - 1) + 24 * (addition_set - 1)

        events.append(Event(EventType.HITSOUND, hitsound_idx))
        events.append(Event(EventType.VOLUME, volume))
        event_times.append(group_time)
        event_times.append(group_time)

    def _clip_dist(self, dist: int) -> int:
        """Clip distance to valid range."""
        return int(np.clip(dist, self.dist_min, self.dist_max))

    def _scale_clip_pos(self, pos: npt.NDArray) -> Tuple[int, int]:
        """Clip position to valid range."""
        p = pos / self.position_precision
        return (int(np.clip(p[0], self.x_min / self.position_precision, self.x_max / self.position_precision)),
                int(np.clip(p[1], self.y_min / self.position_precision, self.y_max / self.position_precision)))

    def _add_position_event(self, pos: npt.NDArray, last_pos: npt.NDArray, time: timedelta, events: list[Event], event_times: list[int]) -> npt.NDArray:
        time_ms = int(time.total_seconds() * 1000)
        if self.add_distances:
            dist = self._clip_dist(np.linalg.norm(pos - last_pos))
            events.append(Event(EventType.DISTANCE, dist))
            event_times.append(time_ms)

        if self.add_positions:
            p = self._scale_clip_pos(pos)
            if self.position_split_axes:
                events.append(Event(EventType.POS_X, p[0]))
                events.append(Event(EventType.POS_Y, p[1]))
                event_times.append(time_ms)
                event_times.append(time_ms)
            else:
                events.append(Event(EventType.POS, int((p[0] - self.x_min / self.position_precision) +
                                                       (p[1] - self.y_min / self.position_precision) *
                                                       ((self.x_max - self.x_min) / self.position_precision + 1))
                                    ))
                event_times.append(time_ms)

        return pos

    def _add_mania_column_event(self, pos: npt.NDArray, columns: int, time: timedelta, events: list[Event], event_times: list[int]) -> None:
        time_ms = int(time.total_seconds() * 1000)
        column = int(np.clip(pos[0] / 512 * columns, 0, columns - 1))
        events.append(Event(EventType.MANIA_COLUMN, column))
        event_times.append(time_ms)

    def _add_group(
            self,
            event: EventType | Event,
            time: timedelta,
            events: list[Event],
            event_times: list[int],
            beatmap: Beatmap,
            *,
            time_event: bool = False,
            add_snap=True,
            pos: npt.NDArray = None,
            last_pos: npt.NDArray = None,
            new_combo: bool = False,
            hitsound_ref_times: list[timedelta] = None,
            hitsounds: list[int] = None,
            additions: list[str] = None,
            scroll_speed: Optional[float] = None,
    ) -> npt.NDArray:
        """Add a group of events to the event list."""
        time_ms = int(time.total_seconds() * 1000) if time is not None else None

        if isinstance(event, EventType):
            event = Event(event)

        if self.types_first:
            events.append(event)
            event_times.append(time_ms)
        if time_event:
            self._add_time_event(time, beatmap, events, event_times, add_snap)
        if pos is not None:
            if beatmap.mode in [0, 2]:
                last_pos = self._add_position_event(pos, last_pos, time, events, event_times)
            elif beatmap.mode == 3:
                self._add_mania_column_event(pos, int(beatmap.circle_size), time, events, event_times)
        if new_combo and beatmap.mode in [0, 2]:
            events.append(Event(EventType.NEW_COMBO))
            event_times.append(time_ms)
        if scroll_speed is not None:
            events.append(Event(EventType.SCROLL_SPEED, round(np.clip(scroll_speed, 0.01, 10) * 100)))
            event_times.append(time_ms)
        if hitsound_ref_times is not None:
            for i, ref_time in enumerate(hitsound_ref_times):
                self._add_hitsound_event(ref_time, time_ms, hitsounds[i], additions[i], beatmap, events, event_times)
        if not self.types_first:
            events.append(event)
            event_times.append(time_ms)

        return last_pos

    def _parse_circle(self, circle: Circle, events: list[Event], event_times: list[int], last_pos: npt.NDArray, beatmap: Beatmap) -> npt.NDArray:
        """Parse a circle hit object.

        Args:
            circle: Circle object.
            events: List of events to add to.
            last_pos: Last position of the hit objects.

        Returns:
            pos: Position of the circle.
        """
        return self._add_group(
            EventType.CIRCLE,
            circle.time,
            events,
            event_times,
            beatmap,
            time_event=True,
            pos=np.array(circle.position),
            last_pos=last_pos,
            new_combo=circle.new_combo,
            hitsound_ref_times=[circle.time],
            hitsounds=[circle.hitsound],
            additions=[circle.addition],
            scroll_speed=self.scroll_speed_at(circle.time, beatmap) if beatmap.mode == 1 else None,
        )

    def _parse_slider(self, slider: Slider, events: list[Event], event_times: list[int], last_pos: npt.NDArray, beatmap: Beatmap) -> npt.NDArray:
        """Parse a slider hit object.

        Args:
            slider: Slider object.
            events: List of events to add to.
            last_pos: Last position of the hit objects.

        Returns:
            pos: Last position of the slider.
        """
        # Ignore sliders which are too big
        if len(slider.curve.points) >= 100:
            return last_pos

        last_pos = self._add_group(
            EventType.SLIDER_HEAD,
            slider.time,
            events,
            event_times,
            beatmap,
            time_event=True,
            pos=np.array(slider.position),
            last_pos=last_pos,
            new_combo=slider.new_combo,
            hitsound_ref_times=[slider.time],
            hitsounds=[slider.edge_sounds[0] if len(slider.edge_sounds) > 0 else 0],
            additions=[slider.edge_additions[0] if len(slider.edge_additions) > 0 else '0:0'],
            scroll_speed=self.scroll_speed_at(slider.time, beatmap) if self.add_sv else None,
        )

        duration: timedelta = (slider.end_time - slider.time) / slider.repeat
        control_point_count = len(slider.curve.points)

        def append_control_points(event_type: EventType, last_pos: npt.NDArray = last_pos) -> npt.NDArray:
            for i in range(1, control_point_count - 1):
                last_pos = add_anchor(event_type, i, last_pos)

            return last_pos

        def add_anchor(event_type: EventType, i: int, last_pos: npt.NDArray) -> npt.NDArray:
            return self._add_group(
                event_type,
                slider.time + i / (control_point_count - 1) * duration,
                events,
                event_times,
                beatmap,
                pos=np.array(slider.curve.points[i]),
                last_pos=last_pos,
            )

        if isinstance(slider.curve, Linear):
            last_pos = append_control_points(EventType.RED_ANCHOR, last_pos)
        elif isinstance(slider.curve, Catmull):
            last_pos = append_control_points(EventType.CATMULL_ANCHOR, last_pos)
        elif isinstance(slider.curve, Perfect):
            last_pos = append_control_points(EventType.PERFECT_ANCHOR, last_pos)
        elif isinstance(slider.curve, MultiBezier):
            for i in range(1, control_point_count - 1):
                if slider.curve.points[i] == slider.curve.points[i + 1]:
                    last_pos = add_anchor(EventType.RED_ANCHOR, i, last_pos)
                elif slider.curve.points[i] != slider.curve.points[i - 1]:
                    last_pos = add_anchor(EventType.BEZIER_ANCHOR, i, last_pos)

        # Add body hitsounds and remaining edge hitsounds
        last_pos = self._add_group(
            EventType.LAST_ANCHOR,
            slider.time + duration,
            events,
            event_times,
            beatmap,
            time_event=True,
            pos=np.array(slider.curve.points[-1]),
            last_pos=last_pos,
            hitsound_ref_times=[slider.time + timedelta(milliseconds=1)] + [slider.time + i * duration for i in range(1, slider.repeat)],
            hitsounds=[slider.hitsound] + [slider.edge_sounds[i] if len(slider.edge_sounds) > i else 0 for i in range(1, slider.repeat)],
            additions=[slider.addition] + [slider.edge_additions[i] if len(slider.edge_additions) > i else '0:0' for i in range(1, slider.repeat)],
        )

        return self._add_group(
            EventType.SLIDER_END,
            slider.end_time,
            events,
            event_times,
            beatmap,
            time_event=True,
            pos=np.array(slider.curve(1)),
            last_pos=last_pos,
            hitsound_ref_times=[slider.end_time],
            hitsounds=[slider.edge_sounds[-1] if len(slider.edge_sounds) > 0 else 0],
            additions=[slider.edge_additions[-1] if len(slider.edge_additions) > 0 else '0:0'],
        )

    def _parse_spinner(self, spinner: Spinner, events: list[Event], event_times: list[int], beatmap: Beatmap) -> npt.NDArray:
        """Parse a spinner hit object.

        Args:
            spinner: Spinner object.
            events: List of events to add to.

        Returns:
            pos: Last position of the spinner.
        """
        self._add_group(
            EventType.SPINNER,
            spinner.time,
            events,
            event_times,
            beatmap,
            time_event=True,
        )

        self._add_group(
            EventType.SPINNER_END,
            spinner.end_time,
            events,
            event_times,
            beatmap,
            time_event=True,
            hitsound_ref_times=[spinner.end_time],
            hitsounds=[spinner.hitsound],
            additions=[spinner.addition],
        )

        return np.array((256, 192))

    def _parse_hold_note(self, hold_note: HoldNote, events: list[Event], event_times: list[int], beatmap: Beatmap) -> npt.NDArray:
        """Parse a hold note hit object.

        Args:
            hold note: Hold note object.
            events: List of events to add to.

        Returns:
            pos: Last position of the spinner.
        """
        pos = np.array(hold_note.position)

        self._add_group(
            EventType.HOLD_NOTE,
            hold_note.time,
            events,
            event_times,
            beatmap,
            time_event=True,
            pos=pos,
            hitsound_ref_times=[hold_note.time],
            hitsounds=[hold_note.hitsound],
            additions=[hold_note.addition],
        )

        self._add_group(
            EventType.HOLD_NOTE_END,
            hold_note.end_time,
            events,
            event_times,
            beatmap,
            time_event=True,
            pos=pos,
        )

        return pos

    def _parse_drumroll(self, slider: Slider, events: list[Event], event_times: list[int], beatmap: Beatmap):
        """Parse a drumroll hit object.

        Args:
            slider: Slider object.
            events: List of events to add to.
        """
        self._add_group(
            EventType.DRUMROLL,
            slider.time,
            events,
            event_times,
            beatmap,
            time_event=True,
            hitsound_ref_times=[slider.time],
            hitsounds=[slider.hitsound],  # Edge hitsounds are not supported in drumrolls
            additions=[slider.addition],
            scroll_speed=self.scroll_speed_at(slider.time, beatmap),
        )

        self._add_group(
            EventType.DRUMROLL_END,
            slider.end_time,
            events,
            event_times,
            beatmap,
            time_event=True,
        )

    def _parse_denden(self, spinner: Spinner, events: list[Event], event_times: list[int], beatmap: Beatmap):
        """Parse a denden hit object.

        Args:
            spinner: Spinner object.
            events: List of events to add to.
        """
        self._add_group(
            EventType.DENDEN,
            spinner.time,
            events,
            event_times,
            beatmap,
            time_event=True,
            hitsound_ref_times=[spinner.time],
            hitsounds=[spinner.hitsound],
            additions=[spinner.addition],
            scroll_speed=self.scroll_speed_at(spinner.time, beatmap),
        )

        self._add_group(
            EventType.DENDEN_END,
            spinner.end_time,
            events,
            event_times,
            beatmap,
            time_event=True,
        )
