import slider.utils


def test_accuracy():
    assert slider.utils.accuracy(1, 0, 0, 0) == 1.0
    assert round(slider.utils.accuracy(0, 1, 0, 0), 4) == 0.3333
    assert round(slider.utils.accuracy(0, 0, 1, 0), 4) == 0.1667
    assert slider.utils.accuracy(0, 0, 0, 1) == 0.0
    assert round(slider.utils.accuracy(982, 100, 43, 14), 4) == 0.8977
