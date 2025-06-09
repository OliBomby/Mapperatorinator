from pathlib import Path

from slider import Beatmap


def example_beatmap(name):
    """Load one of the example beatmaps.

    Parameters
    ----------
    name : str
        The name of the example file to open.
    """
    return Beatmap.from_path(Path(__file__).parent / name)


_sendan_life_versions = frozenset({
    "Easy",
    "Normal",
    "Little's Hard",
    "Little's Insane",
    "Extra",
    "Crystal's Garakowa",
})


def sendan_life(version="Crystal's Garakowa"):
    """Load a version of the Sendan Life beatmap.

    Parameters
    ----------
    version : str
        The version to load.

    Returns
    -------
    sendan_life : Beatmap
        The beatmap object.
    """
    if version not in _sendan_life_versions:
        raise ValueError(
            f'unknown version {version}, options: {set(_sendan_life_versions)}'
        )

    return example_beatmap(
        f'Remo Prototype[CV Hanamori Yumiri] - Sendan Life (Narcissu)'
        f' [{version}].osu'
    )


_ai_no_scenario_versions = frozenset({
    "Beginner",
    "Extra",
    "Hard",
    "ktgster's Insane",
    "Kyshiro's Extra",
    "Nathan's Insane",
    "Normal",
    "pishi's Extra",
    "Sharkie's Insane",
    "sheela's Very Hard",
    "Smoothie World's Extra",
    "Super Beginner",
    "Tatoe",
    "toybot's Insane",
    "Ultra Beginner",
    "Walao's Advanced",
    "Yuistrata's Easy",
})


def miiro_vs_ai_no_scenario(version='Tatoe'):
    """Load a version of the MIIRO vs. Ai no Scenario beatmap.

    Parameters
    ----------
    version : str
        The version to load.

    Returns
    -------
    sendan_life : Beatmap
        The beatmap object.
    """
    if version not in _ai_no_scenario_versions:
        raise ValueError(
            f'unknown version {version}, options:'
            f' {set(_ai_no_scenario_versions)}'
        )

    return example_beatmap(
        f'AKINO from bless4 & CHiCO with HoneyWorks - MIIRO vs. Ai no Scenario'
        f' (monstrata) [{version}].osu'
    )
