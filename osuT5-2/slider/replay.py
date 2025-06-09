import bisect
import datetime
from enum import unique
import os
import lzma

from .beatmap import Circle, Slider, Spinner
from .bit_enum import BitEnum
from .game_mode import GameMode
from .mod import Mod, od_to_ms, circle_radius
from .position import Position
from .utils import (accuracy, lazyval, orange, consume_byte, consume_short,
                    consume_int, consume_string, consume_datetime)


@unique
class ActionBitMask(BitEnum):
    """The bitmask values for the action type.
    """
    m1 = 1
    m2 = 2
    k1 = 5
    k2 = 10


class Action:
    """A user action.

    Parameters
    ----------
    offset : timedelta
        The offset since the beginning of the song.
    position : Position
        The position of the cursor.
    key1 : bool
        Is the first keyboard key pressed?
    key2 : bool
        Is the second keyboard key pressed?
    mouse1 : bool
        Is the first mouse button pressed?
    mouse2 : bool
        is the second mouse button pressed?
    """
    def __init__(self, offset, position, key1, key2, mouse1, mouse2):
        self.offset = offset
        self.position = position
        self.key1 = key1
        self.key2 = key2
        self.mouse1 = mouse1
        self.mouse2 = mouse2

    @property
    def action_bitmask(self):
        """Get the action bitmask from an action.
        """
        return ActionBitMask.pack(
            m1=self.mouse1,
            m2=self.mouse2,
            k1=self.key1,
            k2=self.key2,
        )

    def __repr__(self):
        actions = []
        if self.key1:
            actions.append("K1")
        if self.key2:
            actions.append("K2")
        if self.mouse1:
            actions.append("M1")
        if self.mouse2:
            actions.append("M2")
        return (f"<{type(self).__qualname__}: {self.offset}, {self.position}, "
                f"{' + '.join(actions) or 'No Keypresses'}>")


def _consume_life_bar_graph(buffer):
    life_bar_graph_raw = consume_string(buffer)
    return [
        (datetime.timedelta(milliseconds=int(offset)), float(value))
        for offset, value in (
            pair.split('|') for pair in life_bar_graph_raw.split(',') if pair
        )
    ]


def _consume_actions(buffer):
    compressed_byte_count = consume_int(buffer)
    compressed_data = buffer[:compressed_byte_count]
    del buffer[:compressed_byte_count]
    decompressed_data = lzma.decompress(compressed_data)

    out = []
    offset = 0
    for raw_action in decompressed_data.split(b','):
        if not raw_action:
            continue
        raw_offset, x, y, raw_action_mask = raw_action.split(b'|')
        action_mask = ActionBitMask.unpack(int(raw_action_mask))
        offset += int(raw_offset)
        out.append(Action(
            datetime.timedelta(milliseconds=offset),
            Position(float(x), float(y)),
            action_mask['m1'],
            action_mask['m2'],
            action_mask['k1'],
            action_mask['k2'],
        ))
    return out


def _within(p1, p2, d):
    """Determines whether 2 points are within a distance of each other

    Parameters
    ---------
    p1 : Position
        The first point
    p2 : Position
        The second point
    d : int or float
        The distance

    Returns
    ----------
    bool
        Whether the distance between the points is less than d
    """
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 < d ** 2


def _pressed(datum):
    return datum.key1 or datum.key2 or datum.mouse1 or datum.mouse2


def _process_circle(obj, rdatum, hw, scores):
    out_by = abs(rdatum.offset - obj.time)
    if out_by < datetime.timedelta(milliseconds=hw.hit_300):
        scores["300s"].append(obj)
    elif out_by < datetime.timedelta(milliseconds=hw.hit_100):
        scores["100s"].append(obj)
    else:
        # must be within the 50 hit window or we wouldn't be here
        scores["50s"].append(obj)


def _process_slider(obj, rdata, head_hit, rad, scores):
    t_changes = []
    t_changes_append = t_changes.append
    duration = obj.end_time - obj.time

    if head_hit:
        t_changes_append((rdata[0].offset - obj.time) / duration)
        on = True
    else:
        scores["slider_breaks"].append(obj)
        on = False

    for datum in rdata:
        t = (datum.offset - obj.time) / duration
        if 0 <= t <= 1:
            nearest_pos = obj.curve(t)
            if (on and
                not (_pressed(datum)
                     and _within(nearest_pos, datum.position, rad * 2.4))):
                t_changes_append(t)
                on = False
            elif (not on and
                  (_pressed(datum) and
                   _within(nearest_pos, datum.position, rad))):
                t_changes_append(t)
                on = True

    tick_ts = list(orange(obj.tick_rate, obj.num_beats, obj.tick_rate))
    missed_points = 0 if head_hit else 1
    for tick in tick_ts:
        bi = bisect.bisect_left(t_changes, tick)
        if bi % 2 == 0:
            # missed a tick
            if tick is tick_ts[-1]:
                if (len(t_changes) > 0 and
                        len(t_changes) == bi and
                        abs(tick_ts[-1] - t_changes[-1]) < 0.1):
                    # held close enough to last tick
                    continue
                # end tick doesn't cause sliderbreak
            elif obj not in scores["slider_breaks"]:
                scores["slider_breaks"].append(obj)
            missed_points += 1

    if missed_points == obj.ticks:
        # all ticks and head missed -> miss
        scores["misses"].append(obj)
    elif missed_points > obj.ticks / 2:
        scores["50s"].append(obj)
    elif missed_points > 0:
        scores["100s"].append(obj)
    else:
        scores["300s"].append(obj)


class Replay:
    """An osu! replay.

    Parameters
    ----------
    mode : GameMode
        The game mode.
    version : int
        The version of osu! used to create this replay.
    beatmap_md5 : str
        The md5 hash of the beatmap played in this replay.
    player_name : str
        The name of the player who recorded this replay.
    replay_md5 : str
        The md5 hash of part of the data in this replay.
    count_300 : int
        The number of 300's hit in the replay.
    count_100 : int
        The number of 100's hit in the replay.
    count_50 : int
        The number of 50's hit in the replay.
    count_geki : int
        The number of geki in the replay. A geki is when the user scores all
        300's for a given color section.
    count_katu : int
        The number of katu in the replay. A katu is when the user completes
        a given color section without any 50's or misses. All 300's would
        result in a geki instead of a katu.
    count_miss : int
        The number of misses in the replay.
    score : int
        The score earned in this replay. This is the normal score, not
        performance points.
    max_combo : int
        The largest combo achieved in this replay.
    full_combo : bool
        Did the player earn a max combo in this replay?
    no_fail : bool
        Was the no_fail mod used?
    easy : bool
        Was the easy mod used?
    no_video : bool
        Was the no_video mod used?
    hidden : bool
        Was the hidden mod used?
    hard_rock : hard_rock
        Was the hard_rock mod used?
    sudden_death : bool
        Was the sudden_death mod used?
    double_time : bool
        Was the double_time mod used?
    relax : bool
        Was the relax mod used?
    half_time : bool
        Was the half_time mod used?
    nightcore : bool
        Was the nightcore mod used?
    flashlight : bool
        Was the flashlight mod used?
    autoplay : bool
        Was the autoplay mod used?
    spun_out : bool
        Was the spun_out mod used?
    auto_pilot : bool
        Was the auto_pilot mod used?
    perfect : bool
        Was the perfect mod used?
    key4 : bool
        Was the key4 mod used?
    key5 : bool
        Was the key5 mod used?
    key6 : bool
        Was the key6 mod used?
    key7 : bool
        Was the key7 mod used?
    key8 : bool
        Was the key8 mod used?
    fade_in : bool
        Was the fade_in mod used?
    random : bool
        Was the random mod used?
    cinema : bool
        Was the cinema mod used?
    target_practice : bool
        Was the target_practice mod used?
    key9 : bool
        Was the key9 mod used?
    coop : bool
        Was the coop mod used?
    key1 : bool
        Was the key1 mod used?
    key3 : bool
        Was the key3 mod used?
    key2 : bool
        Was the key2 mod used?
    scoreV2 : bool
        Was the scoreV2 mod used?
    life_bar_graph : list[timedelta, float]
        A list of time points paired with the value of the life bar at that
        time. These appear in sorted order. The values are in the range [0, 1].
    timestamp : datetime
        When this replay was created.
    actions : list[Action]
        A sorted list of all of the actions recorded from the player.
    beatmap : Beatmap or None
        The beatmap played in this replay if known, otherwise None.
    """
    def __init__(self,
                 mode,
                 version,
                 beatmap_md5,
                 player_name,
                 replay_md5,
                 count_300,
                 count_100,
                 count_50,
                 count_geki,
                 count_katu,
                 count_miss,
                 score,
                 max_combo,
                 full_combo,
                 no_fail,
                 easy,
                 no_video,
                 hidden,
                 hard_rock,
                 sudden_death,
                 double_time,
                 relax,
                 half_time,
                 nightcore,
                 flashlight,
                 autoplay,
                 spun_out,
                 auto_pilot,
                 perfect,
                 key4,
                 key5,
                 key6,
                 key7,
                 key8,
                 fade_in,
                 random,
                 cinema,
                 target_practice,
                 key9,
                 coop,
                 key1,
                 key3,
                 key2,
                 scoreV2,
                 life_bar_graph,
                 timestamp,
                 actions,
                 beatmap):
        self.mode = mode
        self.version = version
        self.beatmap_md5 = beatmap_md5
        self.player_name = player_name
        self.replay_md5 = replay_md5
        self.count_300 = count_300
        self.count_100 = count_100
        self.count_50 = count_50
        self.count_geki = count_geki
        self.count_katu = count_katu
        self.count_miss = count_miss
        self.score = score
        self.max_combo = max_combo
        self.full_combo = full_combo
        self.no_fail = no_fail
        self.easy = easy
        self.no_video = no_video
        self.hidden = hidden
        self.hard_rock = hard_rock
        self.sudden_death = sudden_death
        self.double_time = double_time
        self.relax = relax
        self.half_time = half_time
        self.nightcore = nightcore
        self.flashlight = flashlight
        self.autoplay = autoplay
        self.spun_out = spun_out
        self.auto_pilot = auto_pilot
        self.perfect = perfect
        self.key4 = key4
        self.key5 = key5
        self.key6 = key6
        self.key7 = key7
        self.key8 = key8
        self.fade_in = fade_in
        self.random = random
        self.cinema = cinema
        self.target_practice = target_practice
        self.key9 = key9
        self.coop = coop
        self.key1 = key1
        self.key3 = key3
        self.key2 = key2
        self.scoreV2 = scoreV2
        self.life_bar_graph = life_bar_graph
        self.timestamp = timestamp
        self.actions = actions
        self.beatmap = beatmap

    @lazyval
    def accuracy(self):
        """The accuracy achieved in the replay in the range [0, 1].
        """
        if self.mode != GameMode.standard:
            raise NotImplementedError(
                'accuracy for non osu!standard replays is not yet supported',
            )

        return accuracy(
            self.count_300,
            self.count_100,
            self.count_50,
            self.count_miss,
        )

    @lazyval
    def performance_points(self):
        return self.beatmap.performance_points(
            count_300=self.count_300,
            count_100=self.count_100,
            count_50=self.count_50,
            count_miss=self.count_miss,
            easy=self.easy,
            hard_rock=self.hard_rock,
            half_time=self.half_time,
            double_time=self.double_time,
            hidden=self.hidden,
            flashlight=self.flashlight,
            spun_out=self.spun_out,
        )

    @lazyval
    def failed(self):
        """Did the user fail this attempt?
        """
        for _, value in self.life_bar_graph:
            if not value:
                return True

        return False

    @classmethod
    def from_path(cls,
                  path,
                  *,
                  library=None,
                  client=None,
                  save=False,
                  retrieve_beatmap=True):
        """Read in a ``Replay`` object from a ``.osr`` file on disk.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the file to read from.
        library : Library, optional
            The library of beatmaps.
        client : Client, optional.
            The client used to find the beatmap.
        save : bool, optional
            If the beatmap needs to be downloaded with the client, should it be
            saved to disk?
        retrieve_beatmap : bool, optional
            Whether to retrieve the beatmap the replay is for.

        Returns
        -------
        replay : Replay
            The parsed replay object.

        Raises
        ------
        ValueError
            Raised when the file cannot be parsed as an ``.osr`` file.
        """
        with open(path, 'rb') as f:
            return cls.from_file(
                f,
                library=library,
                client=client,
                save=save,
                retrieve_beatmap=retrieve_beatmap,
            )

    @classmethod
    def from_directory(cls,
                       path,
                       *,
                       library=None,
                       client=None,
                       save=False,
                       retrieve_beatmap=True):
        """Read in a list of ``Replay`` objects from a directory of ``.osr``
        files.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the directory to read from.
        library : Library, optional
            The library of beatmaps.
        client : Client, optional.
            The client used to find the beatmap.
        save : bool, optional
            If the beatmap needs to be downloaded with the client, should it be
            saved to disk?
        retrieve_beatmap : bool, optional
            Whether to retrieve the beatmap the replay is for.

        Returns
        -------
        replays : list[Replay]
            The parsed replay objects.

        Raises
        ------
        ValueError
            Raised when any file cannot be parsed as an ``.osr`` file.
        """
        return [
            cls.from_path(
                p,
                library=library,
                client=client,
                save=save,
                retrieve_beatmap=retrieve_beatmap,
            )
            for p in os.scandir(path)
            if p.name.endswith('.osr')
        ]

    @classmethod
    def from_file(cls,
                  file,
                  *,
                  library=None,
                  client=None,
                  save=False,
                  retrieve_beatmap=True):
        """Read in a ``Replay`` object from an open file object.

        Parameters
        ----------
        file : file-like
            The file object to read from.
        library : Library, optional
            The library of beatmaps.
        client : Client, optional.
            The client used to find the beatmap.
        save : bool, optional
            If the beatmap needs to be downloaded with the client, should it be
            saved to disk?
        retrieve_beatmap : bool, optional
            Whether to retrieve the beatmap the replay is for.

        Returns
        -------
        replay : Replay
            The parsed replay object.

        Raises
        ------
        ValueError
            Raised when the file cannot be parsed as a ``.osr`` file.
        """
        return cls.parse(
            file.read(),
            library=library,
            client=client,
            save=save,
            retrieve_beatmap=retrieve_beatmap
        )

    @classmethod
    def parse(cls,
              data,
              *,
              library=None,
              client=None,
              save=False,
              retrieve_beatmap=True):
        """Parse a replay from ``.osr`` file data.

        Parameters
        ----------
        data : bytes
            The data from an ``.osr`` file.
        library : Library, optional
            The library of beatmaps.
        client : Client, optional.
            The client used to find the beatmap.
        save : bool, optional
            If the beatmap needs to be downloaded with the client, should it be
            saved to disk?
        retrieve_beatmap : bool, optional
            Whether to retrieve the beatmap the replay is for.

        Returns
        -------
        replay : Replay
            The parsed replay.

        Raises
        ------
        ValueError
            Raised when ``data`` is not in the ``.osr`` format.
        """
        if retrieve_beatmap:
            if library is None and client is None:
                raise ValueError(
                    'one of library or client must be passed if you wish the'
                    ' beatmap to be retrieved',
                )

            use_client = client is not None
            if use_client:
                if library is not None:
                    raise ValueError(
                        'only one of library or client can be passed'
                    )
                library = client.library

        buffer = bytearray(data)

        mode = GameMode(consume_byte(buffer))
        version = consume_int(buffer)
        beatmap_md5 = consume_string(buffer)
        player_name = consume_string(buffer)
        replay_md5 = consume_string(buffer)
        count_300 = consume_short(buffer)
        count_100 = consume_short(buffer)
        count_50 = consume_short(buffer)
        count_geki = consume_short(buffer)
        count_katu = consume_short(buffer)
        count_miss = consume_short(buffer)
        score = consume_int(buffer)
        max_combo = consume_short(buffer)
        full_combo = bool(consume_byte(buffer))
        mod_mask = consume_int(buffer)
        life_bar_graph = _consume_life_bar_graph(buffer)
        timestamp = consume_datetime(buffer)
        actions = _consume_actions(buffer)

        mod_kwargs = Mod.unpack(mod_mask)
        # delete the alias field names
        del mod_kwargs['relax2']
        del mod_kwargs['last_mod']

        if retrieve_beatmap:
            try:
                beatmap = library.lookup_by_md5(beatmap_md5)
            except KeyError:
                if not use_client:
                    raise
                beatmap = client.beatmap(
                    beatmap_md5=beatmap_md5,
                ).beatmap(save=save)
        else:
            beatmap = None

        return cls(
            mode=mode,
            version=version,
            beatmap_md5=beatmap_md5,
            player_name=player_name,
            replay_md5=replay_md5,
            count_300=count_300,
            count_100=count_100,
            count_50=count_50,
            count_geki=count_geki,
            count_katu=count_katu,
            count_miss=count_miss,
            score=score,
            max_combo=max_combo,
            full_combo=full_combo,
            life_bar_graph=life_bar_graph,
            timestamp=timestamp,
            actions=actions,
            beatmap=beatmap,
            **mod_kwargs,
        )

    @lazyval
    def hits(self):
        """Dictionary containing beatmap's hit objects sorted into
        300s, 100s, 50s, misses, slider_breaks as they were hit in the replay

        Each hit object will be in exactly one category except sliders which
        may be in slider_breaks in addition to another category

        Slider calculations are unreliable so some objects may in the wrong
        category.
        Spinners are not yet calculated so are always in the 300s category.
        """
        beatmap = self.beatmap
        actions = self.actions
        scores = {"300s": [],
                  "100s": [],
                  "50s": [],
                  "misses": [],
                  "slider_breaks": [],
                  }
        hw = od_to_ms(beatmap.od(easy=self.easy, hard_rock=self.hard_rock))
        rad = circle_radius(
            beatmap.cs(easy=self.easy, hard_rock=self.hard_rock),
        )
        hit_50_threshold = datetime.timedelta(milliseconds=hw.hit_50)
        i = 0
        for obj in beatmap.hit_objects():
            if self.hard_rock:
                obj = obj.hard_rock
            if isinstance(obj, Spinner):
                # spinners are hard
                scores['300s'].append(obj)
                continue
            # we can ignore events before the hit window so iterate
            # until we get past the beginning of the hit window
            while actions[i].offset < obj.time - hit_50_threshold:
                i += 1
            starti = i
            while actions[i].offset < obj.time + hit_50_threshold:
                if (((actions[i].key1 and not actions[i - 1].key1)
                        or (actions[i].key2 and not actions[i - 1].key2))
                        and _within(actions[i].position, obj.position, rad)):
                    # key pressed that wasn't before and
                    # event is in hit window and correct location
                    if isinstance(obj, Circle):
                        _process_circle(obj, actions[i], hw, scores)
                    elif isinstance(obj, Slider):
                        # Head was hit
                        starti = i
                        while actions[i].offset <= obj.end_time:
                            i += 1
                        _process_slider(
                            obj, actions[starti:i + 1], True, rad, scores
                        )
                    break
                i += 1
            else:
                # no events in the hit window were in the correct location
                if isinstance(obj, Slider):
                    # Slider ticks might still be hit
                    while actions[i].offset <= obj.end_time:
                        i += 1
                    _process_slider(
                        obj, actions[starti:i + 1], False, rad, scores
                    )
                else:
                    scores["misses"].append(obj)
            i += 1
        return scores

    def __repr__(self):
        try:
            accuracy = f'{self.accuracy * 100:.2f}'
        except NotImplementedError:
            accuracy = '<unknown>'

        beatmap = self.beatmap
        if beatmap is None:
            beatmap = '<unknown>'

        return (
            f'<{type(self).__qualname__}: {accuracy}% ('
            f'{self.count_300}/{self.count_100}/'
            f'{self.count_50}/{self.count_miss}) on {beatmap}>'
        )
