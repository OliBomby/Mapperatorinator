from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    SNAPPING = "snap"
    DISTANCE = "dist"
    NEW_COMBO = "new_combo"
    HITSOUND = "hitsound"
    VOLUME = "volume"
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    BEZIER_ANCHOR = "bezier_anchor"
    PERFECT_ANCHOR = "perfect_anchor"
    CATMULL_ANCHOR = "catmull_anchor"
    RED_ANCHOR = "red_anchor"
    LAST_ANCHOR = "last_anchor"
    SLIDER_END = "slider_end"
    BEAT = "beat"
    MEASURE = "measure"
    STYLE = "style"
    DIFFICULTY = "difficulty"
    MAPPER = "mapper"
    DESCRIPTOR = "descriptor"
    POS_X = "pos_x"
    POS_Y = "pos_y"
    POS = "pos"


class ContextType(Enum):
    NONE = "none"
    TIMING = "timing"
    NO_HS = "no_hs"
    GD = "gd"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"

    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"
