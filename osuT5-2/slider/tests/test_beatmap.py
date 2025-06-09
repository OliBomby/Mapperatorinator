import pytest

import slider.example_data.beatmaps
import slider.beatmap
import slider.curve
from slider.position import Position
from datetime import timedelta
from math import isclose


@pytest.fixture
def beatmap():
    return slider.example_data.beatmaps.miiro_vs_ai_no_scenario('Tatoe')


def test_parse_beatmap_format_v3():
    # v3 is a very old beatmap version. We just want to make sure it doesn't
    # error, see #79 and #87 on github.
    slider.example_data.beatmaps.example_beatmap(
        "Sambomaster - Sekai wa Sore wo Ai to Yobunda ze (ZZT the Fifth) "
        "[Normal].osu"
    )


def test_version(beatmap):
    assert beatmap.format_version == 14


def test_display_name(beatmap):
    assert beatmap.display_name == (
        'AKINO from bless4 & CHiCO with HoneyWorks - MIIRO '
        'vs. Ai no Scenario [Tatoe]'
    )


def test_parse_section_general(beatmap):
    assert beatmap.audio_filename == "tatoe.mp3"
    assert beatmap.audio_lead_in == timedelta()
    assert beatmap.preview_time == timedelta(milliseconds=6538)
    assert not beatmap.countdown
    assert beatmap.sample_set == "Normal"
    assert beatmap.stack_leniency == 0.7
    assert beatmap.mode == 0
    assert not beatmap.letterbox_in_breaks
    assert not beatmap.widescreen_storyboard


def test_parse_section_editor(beatmap):
    assert beatmap.distance_spacing == 1.1
    assert beatmap.beat_divisor == 6
    assert beatmap.grid_size == 4
    assert beatmap.timeline_zoom == 1.8


def test_parse_section_metadata(beatmap):
    assert beatmap.title == "MIIRO vs. Ai no Scenario"
    assert beatmap.title_unicode == "海色 vs. アイのシナリオ"
    assert beatmap.artist == "AKINO from bless4 & CHiCO with HoneyWorks"
    assert beatmap.artist_unicode == (
        "AKINO from bless4 & CHiCO with HoneyWorks"
    )
    assert beatmap.creator == "monstrata"
    assert beatmap.version == "Tatoe"
    assert beatmap.source == ""
    assert beatmap.tags == [
        'kyshiro',
        'sukinathan',
        'ktgster',
        'pishifat',
        'smoothie',
        'world',
        'walaowey',
        'toybot',
        'sheela901',
        'yuii-',
        'Sharkie',
        'みいろ',
        'tv',
        'size',
        'opening',
        'kantai',
        'collection',
        'kancolle',
        'fleet',
        'girls',
        'magic',
        'kaito',
        '1412',
        'まじっく快斗1412',
        '艦隊これくしょん',
        '-艦これ-'
    ]
    assert beatmap.beatmap_id == 735272
    assert beatmap.beatmap_set_id == 325158


def test_parse_section_difficulty(beatmap):
    assert beatmap.hp_drain_rate == 6.5
    assert beatmap.circle_size == 4
    assert beatmap.overall_difficulty == 9
    assert beatmap.approach_rate == 9.5
    assert beatmap.slider_multiplier == 1.8
    assert beatmap.slider_tick_rate == 1


def test_parse_section_timing_points(beatmap):
    # currently only checking the first timing point
    timing_points_0 = beatmap.timing_points[0]
    assert timing_points_0.offset == timedelta()
    assert isclose(timing_points_0.ms_per_beat, 307.692307692308)
    assert timing_points_0.meter == 4
    # sample_set and sample_type omitted, see #56
    assert timing_points_0.volume == 60
    # inherited is not in class parameter
    assert timing_points_0.kiai_mode == 0


def test_parse_section_hit_objects(beatmap):
    # Only hit object 0 tested for now
    hit_objects_0 = beatmap.hit_objects(stacking=False)[0]
    assert hit_objects_0.position == Position(x=243, y=164)
    assert hit_objects_0.time == timedelta(milliseconds=1076)
    # Hit object note `type` is done by subclassing HitObject
    assert isinstance(hit_objects_0, slider.beatmap.Slider)
    assert hit_objects_0.new_combo
    assert not hit_objects_0.combo_skip
    # Slider specific parameters
    assert hit_objects_0.end_time == timedelta(milliseconds=1178)
    assert hit_objects_0.hitsound == 0
    assert isinstance(hit_objects_0.curve, slider.curve.Linear)
    assert hit_objects_0.curve.points == [Position(x=243, y=164),
                                          Position(x=301, y=175)]
    assert round(hit_objects_0.curve.req_length) == 45
    assert isclose(hit_objects_0.length, 45.0000017166138)
    assert hit_objects_0.ticks == 2
    assert isclose(hit_objects_0.num_beats, 0.3333333460489903)
    assert hit_objects_0.tick_rate == 1.0
    assert isclose(hit_objects_0.ms_per_beat, 307.692307692308)
    assert hit_objects_0.edge_sounds == [2, 0]
    assert hit_objects_0.edge_additions == ['0:0', '0:0']
    assert hit_objects_0.addition == "0:0:0:0:"


def test_hit_objects_stacking():
    hit_objects = [slider.beatmap.Circle(Position(128, 128),
                                         timedelta(milliseconds=x*10),
                                         hitsound=1) for x in range(10)]

    beatmap = slider.Beatmap(
        format_version=14,
        audio_filename="audio.mp3",
        audio_lead_in=timedelta(),
        preview_time=timedelta(),
        countdown=False,
        sample_set="soft",
        stack_leniency=1,
        mode=0,
        letterbox_in_breaks=False,
        widescreen_storyboard=False,
        bookmarks=[0],
        distance_spacing=1,
        beat_divisor=1,
        grid_size=1,
        timeline_zoom=1,
        title="title",
        title_unicode="title",
        artist="artist",
        artist_unicode="artist",
        creator="creator",
        version="1.0",
        source="source",
        tags=["tags"],
        beatmap_id=0,
        beatmap_set_id=0,
        hp_drain_rate=5,
        circle_size=5,
        overall_difficulty=5,
        approach_rate=5,
        slider_multiplier=1,
        slider_tick_rate=1,
        timing_points=[],
        hit_objects=hit_objects
    )
    radius = slider.beatmap.circle_radius(5)
    stack_offset = radius / 10

    for i, ob in enumerate(reversed(beatmap.hit_objects(stacking=True))):
        assert ob.position.y == 128-(i*stack_offset)


def test_hit_objects_hard_rock(beatmap):
    # Only hit object 0 tested for now
    hit_objects_hard_rock_0 = beatmap.hit_objects(hard_rock=True,
                                                  stacking=False)[0]
    assert hit_objects_hard_rock_0.position == Position(x=243, y=220)
    assert hit_objects_hard_rock_0.curve.points == [Position(x=243, y=220),
                                                    Position(x=301, y=209)]


def test_legacy_slider_end():
    beatmap = slider.example_data.beatmaps.miiro_vs_ai_no_scenario()

    # lazer uses float values for the duration of sliders instead of ints as in
    # this library. This means we'll have some rounding errors against the
    # expected position of the last tick. `leniency` is the number of pixels of
    # rounding error to allow.
    # See
    # https://github.com/llllllllll/slider/pull/106#issuecomment-1399583672.
    def test_slider(slider_, expected_last_tick_pos, end_pos, leniency=2):
        assert isinstance(slider_, slider.beatmap.Slider)
        expected_x = expected_last_tick_pos.x
        expected_y = expected_last_tick_pos.y

        last_tick_true = slider_.true_tick_points[-1]

        # make sure the last tick is where we expect it to be
        assert abs(last_tick_true.x - expected_x) <= leniency
        assert abs(last_tick_true.y - expected_y) <= leniency

        last_tick = slider_.tick_points[-1]

        # Make sure the actual sliderends didnt get changed
        assert abs(last_tick.x - end_pos.x) <= leniency
        assert abs(last_tick.y - end_pos.y) <= leniency

    objects = beatmap.hit_objects()

    slider1 = objects[0]
    # last tick positions from lazer (and then rounding). See
    # https://github.com/llllllllll/slider/pull/106#issuecomment-1399583672.
    expected_last_tick_pos1 = Position(x=271, y=169)
    # actual ending of this slider
    end_pos1 = Position(x=287, y=172)

    # check another slider in the map as well (the slider at 20153ms).
    td = timedelta(milliseconds=20153)
    slider2 = beatmap.closest_hitobject(td)
    expected_last_tick_pos2 = Position(x=196, y=110)
    end_pos2 = Position(x=202, y=95)

    test_slider(slider1, expected_last_tick_pos1, end_pos1)
    test_slider(slider2, expected_last_tick_pos2, end_pos2)


def test_closest_hitobject():
    beatmap = slider.example_data.beatmaps.miiro_vs_ai_no_scenario('Beginner')
    hit_object1 = beatmap.hit_objects()[4]
    hit_object2 = beatmap.hit_objects()[5]
    hit_object3 = beatmap.hit_objects()[6]

    middle_t = timedelta(milliseconds=11076 - ((11076 - 9692) / 2))

    assert hit_object1.time == timedelta(milliseconds=8615)
    assert hit_object2.time == timedelta(milliseconds=9692)
    assert hit_object3.time == timedelta(milliseconds=11076)

    assert beatmap.closest_hitobject(timedelta(milliseconds=8615)) == \
        hit_object1
    assert beatmap.closest_hitobject(timedelta(milliseconds=(8615 - 30))) == \
        hit_object1
    assert beatmap.closest_hitobject(middle_t) == hit_object2
    assert beatmap.closest_hitobject(middle_t, side="right") == hit_object3


def test_ar(beatmap):
    assert beatmap.ar() == 9.5


def test_bpm_min(beatmap):
    assert beatmap.bpm_min() == 180


def test_bpm_max(beatmap):
    assert beatmap.bpm_max() == 195


def test_cs(beatmap):
    assert beatmap.cs() == 4


def test_hp(beatmap):
    assert beatmap.hp() == 6.5  # issue #57


def test_od(beatmap):
    assert beatmap.od() == 9


def test_pack(beatmap):
    # Pack the beatmap and parse it again to see if there is difference.
    packed_str = beatmap.pack()
    packed = slider.Beatmap.parse(packed_str)
    # Since sections like Colours and Events are currently omitted by
    # ``Beatmap.parse``, these sections will be missing in .osu files
    # written back from parsed Beatmaps. Fortunately, without these
    # sections, rewritten .osu can still be recognized by osu! client.
    beatmap_attrs = [
        # General section fields
        'audio_filename', 'audio_lead_in', 'preview_time', 'countdown',
        'sample_set', 'stack_leniency', 'mode', 'letterbox_in_breaks',
        'widescreen_storyboard',
        # Editor section fields
        'distance_spacing', 'beat_divisor', 'grid_size', 'timeline_zoom',
        'bookmarks',
        # Metadata section fields
        'title', 'title_unicode', 'artist', 'artist_unicode', 'creator',
        'version', 'source', 'tags', 'beatmap_id', 'beatmap_set_id',
        # Difficulty section fields
        'hp_drain_rate', 'circle_size', 'overall_difficulty', 'approach_rate',
        'slider_multiplier', 'slider_tick_rate',
    ]
    hitobj_attrs = [
        'position', 'time', 'new_combo', 'combo_skip', 'hitsound', 'addition'
    ]
    slider_attrs = [
        'end_time', 'hitsound', 'repeat', 'length', 'ticks', 'num_beats',
        'tick_rate', 'ms_per_beat', 'edge_sounds', 'edge_additions', 'addition'
    ]
    timing_point_attrs = [
        'offset', 'ms_per_beat', 'meter', 'sample_type', 'sample_set',
        'volume', 'kiai_mode'
    ]

    def check_attrs(object1, object2, attr_list):
        for attr in attr_list:
            assert getattr(object1, attr) == getattr(object2, attr)

    def check_curve(curve1, curve2):
        assert type(curve1) is type(curve2)
        assert curve1.req_length == curve2.req_length
        for point1, point2 in zip(curve1.points, curve2.points):
            assert point1 == point2

    check_attrs(beatmap, packed, beatmap_attrs)

    # check hit objects
    assert len(beatmap._hit_objects) == len(packed._hit_objects)
    for hitobj1, hitobj2 in zip(beatmap._hit_objects, packed._hit_objects):
        assert type(hitobj1) is type(hitobj2)
        check_attrs(hitobj1, hitobj2, hitobj_attrs)

        if isinstance(hitobj1, slider.beatmap.Slider):
            check_attrs(hitobj1, hitobj2, slider_attrs)
            check_curve(hitobj1.curve, hitobj2.curve)
        elif isinstance(hitobj1, (slider.beatmap.Spinner,
                                  slider.beatmap.HoldNote)):
            # spinners / hold notes have an additional attribute `end_time`
            assert hitobj1.end_time == hitobj2.end_time
        # circles has no additional attributes beyond `hitobj_attrs`

    # check timing points
    assert len(beatmap.timing_points) == len(packed.timing_points)
    for tp1, tp2 in zip(beatmap.timing_points, packed.timing_points):
        check_attrs(tp1, tp2, timing_point_attrs)
        # make sure both timing points are either inherited or uninherited
        assert (tp1.parent is not None) == (tp2.parent is not None)
