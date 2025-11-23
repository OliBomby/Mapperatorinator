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
    TIMING_POINT = "timing_point"
    GAMEMODE = "gamemode"
    STYLE = "style"
    DIFFICULTY = "difficulty"
    MAPPER = "mapper"
    CS = "cs"
    YEAR = "year"
    HITSOUNDED = "hitsounded"
    SONG_LENGTH = "song_length"
    SONG_POSITION = "song_position"
    GLOBAL_SV = "global_sv"
    MANIA_KEYCOUNT = "keycount"
    HOLD_NOTE_RATIO = "hold_note_ratio"
    SCROLL_SPEED_RATIO = "scroll_speed_ratio"
    DESCRIPTOR = "descriptor"
    POS_X = "pos_x"
    POS_Y = "pos_y"
    POS = "pos"
    KIAI = "kiai"
    MANIA_COLUMN = "column"
    HOLD_NOTE = "hold_note"
    HOLD_NOTE_END = "hold_note_end"
    SCROLL_SPEED_CHANGE = "scroll_speed_change"
    SCROLL_SPEED = "scroll_speed"
    DRUMROLL = "drumroll"
    DRUMROLL_END = "drumroll_end"
    DENDEN = "denden"
    DENDEN_END = "denden_end"
    CONTROL = "control"


class ContextType(Enum):
    NONE = "none"
    TIMING = "timing"
    NO_HS = "no_hs"
    GD = "gd"
    MAP = "map"
    KIAI = "kiai"
    SV = "sv"


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
