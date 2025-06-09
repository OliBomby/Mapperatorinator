from enum import IntEnum, unique


@unique
class GameMode(IntEnum):
    """The various game modes in osu!.
    """
    standard = 0
    taiko = 1
    ctb = 2
    mania = 3
