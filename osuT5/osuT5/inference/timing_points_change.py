from datetime import timedelta
from functools import cmp_to_key

from slider import TimingPoint
import math
from typing import List


def copy(tp: TimingPoint):
    return TimingPoint(tp.offset, tp.ms_per_beat, tp.meter, tp.sample_type, tp.sample_set,
                       tp.volume, tp.parent, tp.kiai_mode)


def same_effect(tp: TimingPoint, other: TimingPoint):
    return (tp.ms_per_beat == other.ms_per_beat and tp.meter == other.meter and tp.sample_type == other.sample_type and
            tp.sample_set == other.sample_set and tp.volume == other.volume and
            ((tp.parent is None) == (other.parent is None)) and tp.kiai_mode == other.kiai_mode)


def compare_timing_point(tp1: TimingPoint, tp2: TimingPoint) -> int:
    if tp1 is tp2:
        return 0
    if tp2 is None:
        return 1
    if tp1 is None:
        return -1

    offset_comparison = (tp1.offset > tp2.offset) - (tp1.offset < tp2.offset)
    if offset_comparison != 0:
        return offset_comparison
    return -(((tp1.parent is None) > (tp2.parent is None)) - ((tp1.parent is None) < (tp2.parent is None)))


def sort_timing_points(timing_points: List[TimingPoint]) -> List[TimingPoint]:
    return sorted(timing_points, key=cmp_to_key(compare_timing_point))


class TimingPointsChange:
    def __init__(self, tp_new: TimingPoint, mpb: bool = False, meter: bool = False, sampleset: bool = False,
                 index: bool = False, volume: bool = False, uninherited: bool = False, kiai: bool = False,
                 fuzzyness: float = 2):
        self.my_tp = tp_new
        self.ms_per_beat = mpb
        self.meter = meter
        self.sample_type = sampleset
        self.sample_set = index
        self.volume = volume
        self.uninherited = uninherited
        self.kiai_mode = kiai
        self.fuzzyness = fuzzyness / 1000

    def add_change(self, timing: List[TimingPoint], all_after: bool = False) -> List[TimingPoint]:
        adding_timing_point = None
        prev_timing_point = None
        on_timing_points = []
        on_has_red = False
        on_has_green = False

        for tp in timing:
            if tp is None:
                continue  # Continue nulls to avoid exceptions
            if tp.offset < self.my_tp.offset and (prev_timing_point is None or tp.offset >= prev_timing_point.offset):
                prev_timing_point = tp
            if math.isclose(tp.offset.total_seconds(), self.my_tp.offset.total_seconds(), abs_tol=self.fuzzyness):
                on_timing_points.append(tp)
                on_has_red = (tp.parent is None) or on_has_red
                on_has_green = (tp.parent is not None) or on_has_green

        if on_timing_points:
            prev_timing_point = on_timing_points[-1]

        if self.uninherited and not on_has_red:
            # Make new redline
            if prev_timing_point is None:
                adding_timing_point = copy(self.my_tp)
                adding_timing_point.parent = None
            else:
                adding_timing_point = copy(prev_timing_point)
                adding_timing_point.offset = self.my_tp.offset
                adding_timing_point.parent = None
            on_timing_points.append(adding_timing_point)

        if not self.uninherited and (not on_timing_points or (self.ms_per_beat and not on_has_green)):
            # Make new greenline (based on prev)
            #
            # Greenlines must always inherit from an actual redline in the timing list. If we're about to place
            # a greenline before any redline exists, create a redline (or move the first redline backwards) so
            # the parent relationship and SV sign are valid.

            # If we don't have any previous timing point, we're inserting at/near the beginning.
            # Ensure a redline exists before this greenline and use it as parent.
            if prev_timing_point is None:
                prev_timing_point = self._ensure_redline_before(self.my_tp.offset, timing)

            # Now create the greenline.
            adding_timing_point = copy(prev_timing_point)
            adding_timing_point.offset = self.my_tp.offset
            adding_timing_point.parent = prev_timing_point if prev_timing_point.parent is None else prev_timing_point.parent
            if prev_timing_point.parent is None:
                adding_timing_point.ms_per_beat = -100
            on_timing_points.append(adding_timing_point)

        for on in on_timing_points:
            if self.ms_per_beat and (self.uninherited == (on.parent is None)):
                on.ms_per_beat = self.my_tp.ms_per_beat
            if self.meter and self.uninherited and on.parent is None:
                on.meter = self.my_tp.meter
            if self.sample_type:
                on.sample_type = self.my_tp.sample_type
            if self.sample_set:
                on.sample_set = self.my_tp.sample_set
            if self.volume:
                on.volume = self.my_tp.volume
            if self.kiai_mode:
                on.kiai_mode = self.my_tp.kiai_mode

        if adding_timing_point and (prev_timing_point is None or not same_effect(adding_timing_point, prev_timing_point) or self.uninherited):
            timing.append(adding_timing_point)

        if all_after:
            # Change every timing point after
            for tp in timing:
                if tp.offset > self.my_tp.offset:
                    if self.sample_type:
                        tp.sample_type = self.my_tp.sample_type
                    if self.sample_set:
                        tp.sample_set = self.my_tp.sample_set
                    if self.volume:
                        tp.volume = self.my_tp.volume
                    if self.kiai_mode:
                        tp.kiai_mode = self.my_tp.kiai_mode

        # Sort all timing points
        return sort_timing_points(timing)

    @staticmethod
    def apply_changes(timing: List[TimingPoint], timing_points_changes: List['TimingPointsChange'], all_after: bool = False) -> List[TimingPoint]:
        timing_points_changes.sort(key=lambda o: o.my_tp.offset)
        for change in timing_points_changes:
            timing = change.add_change(timing, all_after)
        return timing

    def _first_redline(self, tps: List[TimingPoint]):
        reds = [tp for tp in tps if tp is not None and tp.parent is None]
        if not reds:
            return None
        return min(reds, key=lambda t: t.offset)

    def _ensure_redline_before(self, time_offset, tps: List[TimingPoint]) -> TimingPoint:
        """Ensure there is a redline at or before time_offset.

        If the earliest redline exists but is after time_offset, move it backwards by enough measures.
        If there is no redline at all, insert a new default redline.

        Returns the redline to use as parent.
        """
        first = self._first_redline(tps)
        if first is None:
            new_red = TimingPoint(time_offset, 1000, 4, 2, -1, 100, None, False)
            tps.append(new_red)
            return new_red

        if first.offset > time_offset:
            measure_ms = first.ms_per_beat * first.meter
            back_ms = first.offset - time_offset
            n_measures = math.ceil(back_ms / measure_ms)
            first.offset -= timedelta(milliseconds=n_measures * measure_ms)
        return first

    def debug(self):
        print(self.my_tp.__dict__)
        print(f"{self.ms_per_beat}, {self.meter}, {self.sample_type}, {self.sample_set}, {self.volume}, {self.uninherited}, {self.kiai_mode}")
