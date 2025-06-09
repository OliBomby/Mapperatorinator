import datetime
from enum import IntEnum, unique

import requests

from .game_mode import GameMode
from .mod import Mod
from .utils import lazyval, accuracy


@unique
class ApprovedState(IntEnum):
    """The state of a beatmap's approval.
    """
    loved = 4
    qualified = 3
    approved = 2
    ranked = 1
    pending = 0
    WIP = -1
    graveyard = -2


@unique
class Genre(IntEnum):
    """The genres that appear on the osu! website.
    """
    any = 0
    unspecified = 1
    video_game = 2
    anime = 3
    rock = 4
    pop = 5
    other = 6
    novelty = 7
    # note: there is no 8
    hip_hop = 9
    electronic = 10


@unique
class Language(IntEnum):
    """The languages that appear on the osu! website.
    """
    any = 0
    other = 1
    english = 2
    japanese = 3
    chinese = 4
    instrumental = 5
    korean = 6
    french = 7
    german = 8
    swedish = 9
    spanish = 10
    italian = 11


def _beatmap(self, *, save=False):
    """Lookup the associated beatmap object.

    Parameters
    ----------
    save : bool
        If the beatmap is not in the library, should it be saved?

    Returns
    -------
    beatmap : Beatmap
        The associated beatmap object.
    """
    beatmap = self._beatmap
    if beatmap is not None:
        return beatmap

    self._beatmap = beatmap = self._library.lookup_by_id(
        self.beatmap_id,
        download=True,
        save=save,
    )
    return beatmap


_beatmap.__name__ = 'beatmap'


class BeatmapResult:
    """A beatmap as represented by the osu! API.

    Parameters
    ----------
    library : Library
        The library used to store the Beatmap object.
    title : str
        The beatmap's title.
    version : str
        The beatmap's version.
    beatmap_id : int
        The beatmap_id.
    approved : ApprovedState
        The state of the beatmap's approved.
    approved_date : datetime.datetime
        The date when this map was approved.
    last_update : datetime.datetime
        The last date when this map was updated.
    star_rating : float
        The star rating for the song.
    hit_length : datetime.timedelta
        The amount of time from the first element to the last, not counting
        breaks.
    genre : Genre
        The genre that appears on the osu! website.
    language : Language
        The language that appears on the osu! website.
    total_length : datetime.timedelta
        The amount of time from the first element to the last, counting breaks.
    beatmap_md5 : str
        The md5 hash of the beatmap.
    favourite_count : int
        The number of times the beatmap has been favorited.
    play_count : int
        The number of times this beatmap has been played.
    pass_count : int
        The number of times this beatmap has been passed.
    max_combo : int
        The maximum combo that could be achieved on this beatmap.
    """
    def __init__(self,
                 library,
                 title,
                 version,
                 beatmap_id,
                 approved,
                 approved_date,
                 last_update,
                 star_rating,
                 hit_length,
                 genre,
                 language,
                 total_length,
                 beatmap_md5,
                 favourite_count,
                 play_count,
                 pass_count,
                 max_combo):
        self._library = library
        self.title = title
        self.version = version
        self.beatmap_id = beatmap_id
        self.approved = approved
        self.approved_date = approved_date
        self.last_update = last_update
        self.star_rating = star_rating
        self.hit_length = hit_length
        self.genre = genre
        self.language = language
        self.total_length = total_length
        self.beatmap_md5 = beatmap_md5
        self.favourite_count = favourite_count
        self.play_count = play_count
        self.pass_count = pass_count
        self.max_combo = max_combo

        self._beatmap = None

    beatmap = _beatmap

    def __repr__(self):
        return f'<{type(self).__qualname__}: {self.title} [{self.version}]>'


class UserEvent:
    """Recent events for a user.

    Parameters
    -----------
    library : Library
        The library used to store the Beatmap object.
    display_html : str
        The html to display on the osu! site.
    beatmap_id : int
        The beatmap_id of the event.
    beatmapset_id : int
        The beatmapset_id of the event.
    date : datetime.date
        The date of the event.
    epic_factor : int
        How epic was this event.
    """
    def __init__(self,
                 library,
                 display_html,
                 beatmap_id,
                 beatmapset_id,
                 date,
                 epic_factor):
        self._library = library
        self.display_html = display_html
        self.beatmap_id = beatmap_id
        self.beatmapset_id = beatmapset_id
        self.date = date
        self.epic_factor = epic_factor

        self._beatmap = None

    beatmap = _beatmap


class User:
    """Information about an osu! user.

    Parameters
    ----------
    client : Client
        The client needed to make further requests.
    user_id : int
        The user id.
    user_name : str
        The user name.
    count_300 : int
        The total number of 300s ever hit.
    count_100 : int
        The total number of 100s ever hit.
    count_50 : int
        The total number of 50s ever hit.
    play_count : int
        The total number of plays.
    ranked_score : int
        The user's ranked score.
    total_score : int
        The user's total score.
    pp_rank : int
        The user's rank with the PP system.
    level : float
        The user's level.
    pp_raw : float
        The user's total unweighted PP.
    accuracy : float
        The user's ranked accuracy.
    count_ss : int
        The number of SSs scored.
    count_s : int
        The number of Ss scored.
    count_a : int
        The number of As scored.
    country : str
        The country code for the user's home country.
    pp_country_rank : int
        The user's rank with the PP system limited to other players in their
        country.
    events : list[UserEvent]
        Recent user events.
    game_mode : GameMode
        The game mode the user information is for.
    """
    def __init__(self,
                 client,
                 user_id,
                 user_name,
                 count_300,
                 count_100,
                 count_50,
                 play_count,
                 ranked_score,
                 total_score,
                 pp_rank,
                 level,
                 pp_raw,
                 accuracy,
                 count_ss,
                 count_s,
                 count_a,
                 country,
                 pp_country_rank,
                 events,
                 game_mode):
        self._client = client
        self.user_id = user_id
        self.user_name = user_name
        self.count_300 = count_300
        self.count_100 = count_100
        self.count_50 = count_50
        self.play_count = play_count
        self.ranked_score = ranked_score
        self.total_score = total_score
        self.pp_rank = pp_rank
        self.level = level
        self.pp_raw = pp_raw
        self.accuracy = accuracy
        self.count_ss = count_ss
        self.count_s = count_s
        self.count_a = count_a
        self.country = country
        self.pp_country_rank = pp_country_rank
        self.events = events
        self.game_mode = game_mode

    def __repr__(self):
        return (
            f'<{type(self).__qualname__}: {self.user_name}'
            f' ({self.game_mode.name})>'
        )

    def high_scores(self, limit=10):
        """Lookup the user's high scores.

        Parameters
        ----------
        limit : int, optional
            The number of scores to look up.

        Returns
        -------
        high_scores : list[HighScore]
            The user's high scores.
        """
        return self._client.user_best(
            user_id=self.user_id,
            limit=limit,
            game_mode=self.game_mode,
            _user_ob=self,
        )


class HighScore:
    """A high score for a user or beatmap.

    Parameters
    ----------
    client : Client
        The client used to make further requests.
    beatmap_id : int
        The beatmap_id of the map this is a score for.
    score : int
        The score earned in this high score.
    max_combo : int
        The max combo.
    count_300 : int
        The number of 300s in the high score.
    count_100 : int
        The number of 100s in the high score.
    count_50 : int
        The number of 50s in the high score.
    count_miss : int
        The number of misses in the high score.
    count_katu : int
        The number of katu in the high score.
    count_geki : int
        The number of geki in the high score.
    perfect : bool
        Did the user fc the map?
    mods : set[Mod]
        The mods used.
    user_id : int
        The id of the user who earned this high score.
    rank : str
        The letter rank earned. A suffix ``H`` means hidden or flashlight was
        used, like a silver S(S).
    pp : float
        The unweighted PP earned for this high score.
    """
    def __init__(self,
                 client,
                 beatmap_id,
                 score,
                 max_combo,
                 count_300,
                 count_100,
                 count_50,
                 count_miss,
                 count_katu,
                 count_geki,
                 perfect,
                 mods,
                 user_id,
                 date,
                 rank,
                 pp,
                 _user=None):

        self._client = client
        self._library = client.library
        if _user is not None:
            self.user = _user

        self.beatmap_id = beatmap_id
        self.score = score
        self.max_combo = max_combo
        self.count_300 = count_300
        self.count_100 = count_100
        self.count_50 = count_50
        self.count_miss = count_miss
        self.count_katu = count_katu
        self.count_geki = count_geki
        self.perfect = perfect
        self.mods = mods
        self.user_id = user_id
        self.date = date
        self.rank = rank
        self.pp = pp

        self._beatmap = None

    beatmap = _beatmap

    @lazyval
    def user(self):
        return self._client.user(user_id=self.user_id)

    @property
    def accuracy(self):
        return accuracy(
            self.count_300,
            self.count_100,
            self.count_50,
            self.count_miss,
        )

    def __repr__(self):
        return (
            f'<{type(self).__qualname__} user_id={self.user_id};'
            f' beatmap_id=self.beatmap_id>'
        )


class UnknownBeatmap(LookupError):
    """Raised when a beatmap id or md5 is not known.

    Parameters
    ----------
    kind : {'id', 'md5'}
        The kind of identifier used.
    id_ : str
        The unknown beatmap id.
    """
    def __init__(self, kind, id_):
        self.kind = kind
        self.id_ = id_

    def __str__(self):
        return f'no beatmap found that matched {self.kind}: {self.id_}'


class Client:
    """A client for interacting with the osu! rest API.

    Parameters
    ----------
    library : Library
        The library used to look up or cache beatmap objects.
    api_key : str
        The api key to use.
    """
    DEFAULT_API_URL = 'https://osu.ppy.sh/api'

    def __init__(self, library, api_key, api_url=DEFAULT_API_URL):
        self.library = library
        self.api_key = api_key
        self.api_url = api_url

    def copy(self):
        """Create a copy suitable for use in a new thread.

        Returns
        -------
        Client
            The new copy.
        """
        return type(self)(
            library=self.library.copy(),
            api_key=self.api_key,
            api_url=self.api_url,
        )

    @staticmethod
    def _user_and_type(user_name, user_id, *, required):
        """Normalize user_name or user_id into the 'u' and 't' fields.

        Parameters
        ----------
        user_name : str or None
            The user name.
        user_id : int or None
            The user id.
        required : bool
            Is user information required?

        Returns
        -------
        user_info : (any, str) or None
            The user identifier and type or None if not provided and required
            is False.
        """
        if user_name is not None and user_id is not None:
            raise ValueError('only one of user_name or user_id can be passed')

        if user_name is not None:
            user = user_name
            type_ = 'string'
        elif user_id is not None:
            user = user_id
            type_ = 'id'
        elif required:
            raise ValueError('one of user_name or user_id must be passed')
        else:
            return None

        return user, type_

    # differences in the osu! api names and slider's names
    _beatmap_aliases = {
        'beatmapset_id': 'beatmap_set_id',
        'difficultyrating': 'star_rating',
        'diff_size': 'circle_size',
        'diff_overall': 'overall_difficulty',
        'diff_approach': 'approach_rate',
        'diff_drain': 'health_drain',
        'genre_id': 'genre',
        'language_id': 'language',
        'file_md5': 'beatmap_md5',
        'playcount': 'play_count',
        'passcount': 'pass_count',
    }

    def _parse_date(cs):
        return datetime.datetime.strptime(cs, '%Y-%m-%f %H:%M:%S')

    def _parse_timedelta(cs):
        return datetime.timedelta(seconds=int(cs))

    def _parse_optional(class_):
        def func(cs):
            if cs is None:
                return cs
            return class_(cs)
        return func

    def _identity(cs):
        return cs

    _beatmap_conversions = {
        'approved': lambda cs: ApprovedState(int(cs)),
        'approved_date': _parse_date,
        'last_update': _parse_date,
        'beatmap_id': int,
        'star_rating': float,
        'hit_length': _parse_timedelta,
        'genre': lambda cs: Genre(int(cs)),
        'language': lambda cs: Language(int(cs)),
        'total_length': _parse_timedelta,
        'beatmap_md5': _identity,
        'favourite_count': int,
        'play_count': int,
        'pass_count': int,
        'max_combo': _parse_optional(int),
        'title': _identity,
        'version': _identity,
    }

    def beatmap(self,
                *,
                since=None,
                beatmap_set_id=None,
                beatmap_id=None,
                beatmap_md5=None,
                user_id=None,
                user_name=None,
                game_mode=None,
                include_converted_beatmaps=False,
                limit=500):
        """Retrieve information about a beatmap or set of beatmaps from the
        osu! API.

        Parameters
        ----------
        since : datetime.date, optional
            Return all beatmaps ranked since this date.
        beatmap_set_id : str, optional
            Return all beatmaps in this set.
        beatmap_id : str, optional
            Return the single beatmap with the given id.
        beatmap_md5 : str, optional
            Return the single beatmap with the given hash.
        user_id : int, optional
            Return beatmaps made by the given user.
        user_name : str, optional
            Return beatmaps made by the given user.
        game_mode : GameMode, optional
            Return beatmaps for the given game mode.
        include_converted_beatmaps : bool, optional
            Return converted beatmaps. This only applies for non-osu!standard
            game modes.
        limit : int, optional
            The maximum number of beatmaps to return. This must be less than
            500.
        """
        beatmap_identifiers = {
            k: v for k, v in {
                'beatmap_set_id': beatmap_set_id,
                'beatmap_id': beatmap_id,
                'beatmap_md5': beatmap_md5,
            }.items()
            if v is not None
        }
        if len(beatmap_identifiers) > 1:
            raise ValueError(
                f'only one of beatmap_set_id, beatmap_id, or beatmap_md5 can'
                f' be passed, got {beatmap_identifiers!r}',
            )

        if limit > 500:
            raise ValueError('only 500 beatmaps can be requested at one time')

        parameters = {
            'k': self.api_key,
            'a': int(bool(include_converted_beatmaps)),
            'limit': limit,
        }

        if since is not None:
            parameters['since'] = since.isoformat()

        if beatmap_set_id is not None:
            parameters['s'] = beatmap_set_id
        elif beatmap_id is not None:
            parameters['b'] = beatmap_id
        elif beatmap_md5 is not None:
            parameters['h'] = beatmap_md5

        user_info = self._user_and_type(user_name, user_id, required=False)
        if user_info is not None:
            parameters['u'], parameters['t'] = user_info

        if game_mode is not None:
            parameters['m'] = int(game_mode)

        response = requests.get(
            f'{self.api_url}/get_beatmaps',
            params=parameters,
        )
        response.raise_for_status()

        as_dicts = (
            {self._beatmap_aliases.get(k, k): v for k, v in beatmap.items()}
            for beatmap in response.json()
        )
        converted = [
            BeatmapResult(
                library=self.library,
                **{
                    k: self._beatmap_conversions[k](v)
                    for k, v in d.items()
                    if k in self._beatmap_conversions
                },
            )
            for d in as_dicts
        ]

        if beatmap_id is not None or beatmap_md5 is not None:
            try:
                converted, = converted
            except ValueError:
                if beatmap_id is not None:
                    kind = 'id'
                    id_ = beatmap_id
                else:
                    kind = 'md5'
                    id_ = beatmap_md5

                raise UnknownBeatmap(kind, id_)

        return converted

    _user_aliases = {
        'username': 'user_name',
        'count300': 'count_300',
        'count100': 'count_100',
        'count50': 'count_50',
        'playcount': 'play_count',
        'count_rank_ss': 'count_ss',
        'count_rank_s': 'count_s',
        'count_rank_a': 'count_a',
    }

    def _parse_user_events(events, _parse_date=_parse_date):
        out = []
        for event in events:
            # beatmap id can be null if the event is a supporter gift
            if event['beatmap_id'] is not None:
                event['beatmap_id'] = int(event['beatmap_id'])
            event['date'] = _parse_date(event['date'])
            event['epic_factor'] = event.pop('epicfactor')
            out.append(event)

        return out

    _user_conversions = {
        'user_id': int,
        'user_name': _identity,
        # these attributes can be null for users that have never played before,
        # see user 17906393 / #97 on github
        'count_300': _parse_optional(int),
        'count_100': _parse_optional(int),
        'count_50': _parse_optional(int),
        'play_count': _parse_optional(int),
        'ranked_score': _parse_optional(int),
        'total_score': _parse_optional(int),
        'pp_rank': _parse_optional(int),
        'level': _parse_optional(float),
        'pp_raw': _parse_optional(float),
        'accuracy': _parse_optional(float),
        'count_ss': _parse_optional(int),
        'count_s': _parse_optional(int),
        'count_a': _parse_optional(int),
        'country': _identity,
        'pp_country_rank': _parse_optional(int),
        'events': _parse_user_events,
    }

    def user(self,
             *,
             user_name=None,
             user_id=None,
             game_mode=GameMode.standard,
             event_days=1):
        """Retrieve information about a user.

        Parameters
        ----------
        user_name : str
            The name of the user to look up. This cannot be passed with
            ``user_id``.
        user_id : int
            The id of the user to look up. This cannot be passed with
            ``user_name``
        game_mode : GameMode, optional
            The game mode to look up stats for.
        event_days : int, optional
            Max number of days between now and last event date in the range
            [1, 31].

        Returns
        -------
        user : User
            The requested user.
        """
        if user_name is not None and user_id is not None:
            raise ValueError('only one of user_name or user_id can be passed')

        user, type_ = self._user_and_type(user_name, user_id, required=True)

        if not (1 <= event_days <= 31):
            raise ValueError(
                f'event_days must be in range [1, 31], got {event_days!r}',
            )

        response = requests.get(
            self.api_url + '/get_user',
            params={
                'k': self.api_key,
                'u': user,
                't': type_,
                'm': int(game_mode),
                'event_days': event_days,
            },
        )
        response.raise_for_status()

        dict_, = (
            {self._user_aliases.get(k, k): v for k, v in beatmap.items()}
            for beatmap in response.json()
        )

        library = self.library
        events = dict_['events']

        for event in events:
            event['library'] = library

        return User(
            client=self,
            **{
                k: self._user_conversions[k](v)
                for k, v in dict_.items()
                if k in self._user_conversions
            },
            game_mode=game_mode,
        )
        return user

    _user_best_aliases = {
        'username': 'user_name',
        'maxcombo': 'max_combo',
        'count300': 'count_300',
        'count100': 'count_100',
        'count50': 'count_50',
        'countmiss': 'count_miss',
        'countkatu': 'count_katu',
        'countgeki': 'count_geki',
        'enabled_mods': 'mods',
    }

    _user_best_conversions = {
        'beatmap_id': int,
        'score': int,
        'user_name': _identity,
        'max_combo': int,
        'count_300': int,
        'count_100': int,
        'count_50': int,
        'count_miss': int,
        'count_katu': int,
        'count_geki': int,
        'perfect': lambda m: bool(int(m)),
        'mods': lambda m: {v for v in Mod.unpack(int(m)).values() if v},
        'user_id': int,
        'date': _parse_date,
        'rank': _identity,
        'pp': float,
    }

    def user_best(self,
                  *,
                  user_name=None,
                  user_id=None,
                  game_mode=GameMode.standard,
                  limit=10,
                  _user_ob=None):
        """Retrieve information about a user's best scores.

        Parameters
        ----------
        user_name : str
            The name of the user to look up. This cannot be passed with
            ``user_id``.
        user_id : int
            The id of the user to look up. This cannot be passed with
            ``user_name``
        game_mode : GameMode, optional
            The game mode to look up stats for. Defaults to osu! standard.
        limit : int, optional
            The number of results to return in the range [1, 100].

        Returns
        -------
        high_scores : list[HighScore]
            The user's best scores.
        """
        user, type_ = self._user_and_type(user_name, user_id, required=True)

        if not (1 <= limit <= 100):
            raise ValueError(
                f'limit must be in the range [1, 100], got: {limit!r}',
            )

        response = requests.get(
            self.api_url + '/get_user_best',
            params={
                'k': self.api_key,
                'u': user,
                't': type_,
                'm': int(game_mode),
                'limit': limit,
            },
        )
        response.raise_for_status()

        as_dicts = (
            {self._user_best_aliases.get(k, k): v for k, v in beatmap.items()}
            for beatmap in response.json()
        )
        return [
            HighScore(
                client=self,
                **{
                    k: self._user_best_conversions[k](v)
                    for k, v in d.items()
                    if k in self._user_best_conversions
                },
                _user=_user_ob,
            )
            for d in as_dicts
        ]
