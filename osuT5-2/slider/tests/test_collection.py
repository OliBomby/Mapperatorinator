import slider.example_data.collections


def test_collection_simple():
    collection_db = slider.example_data.collections.test_db()
    assert collection_db.version == 20190410
    assert collection_db.num_collections == 2

    col0, col1 = collection_db.collections
    assert col0.num_beatmaps == 3
    assert col0.md5_hashes == [
        "8a67f16f3c440fa3805c14652306dfe8",
        "7b231749a908b6d17163dc3f42143774",
        "2af0646c72dee92c89649620e4cdb162",
    ]

    assert col1.num_beatmaps == 1
    assert col1.md5_hashes == ["637227d9965d5e0ba72c92a27826f5b3"]
