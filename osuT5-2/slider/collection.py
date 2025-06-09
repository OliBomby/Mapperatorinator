from .replay import consume_int, consume_string


class CollectionDB:
    """An osu! ``collection.db`` file.

    Parameters
    ----------
    version : int
        Version number (e.g. 20150203)
    num_collections : int
        Number of collections
    collections : list[Collection]
        List of :class:`~slider.collection.Collection` s
    """
    def __init__(self, version, num_collections, collections):
        self.version = version
        self.num_collections = num_collections
        self.collections = collections

    @classmethod
    def from_path(cls, path):
        """Read in a ``collection.db`` file from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            The path to the file to read from.
        """
        with open(path, 'rb') as f:
            return cls.from_file(f)

    @classmethod
    def from_file(cls, file):
        """Read in a ``collection.db`` file from an open file object.

        Parameters
        ----------
        file : file-like
            The file object to read from.
        """
        return cls.parse(file.read())

    @classmethod
    def parse(cls, data):
        """Parse from ``collection.db`` data.

        Parameters
        ----------
        data : bytes
            The data from a ``collection.db`` file.
        """
        buffer = bytearray(data)

        version = consume_int(buffer)
        num_collections = consume_int(buffer)
        collections = []
        for i in range(num_collections):
            collections.append(Collection.parse(buffer))

        return cls(version, num_collections, collections)


class Collection:
    """An osu! collection. One or more collections are present in a
    `collection.db` file.

    Parameters
    ----------
    name : str
        The collection's name
    num_beatmaps : int
        How many beatmaps the collection contains
    md5_hashes : list[str]
        List of MD5 hashes of each beatmap
    """
    def __init__(self, name, num_beatmaps, md5_hashes):
        self.name = name
        self.num_beatmaps = num_beatmaps
        self.md5_hashes = md5_hashes

    @classmethod
    def parse(cls, buffer):
        """Parse an osu! collection.

        Parameters
        ----------
        buffer : bytearray
            Buffer passed in from parsing ``CollectionDB``
        """
        name = consume_string(buffer)
        num_beatmaps = consume_int(buffer)
        md5_hashes = []
        for i in range(num_beatmaps):
            md5_hashes.append(consume_string(buffer))

        return cls(name, num_beatmaps, md5_hashes)
