import enum
from functools import reduce
import operator as op


class BitEnum(enum.IntEnum):
    """A type for enums representing bitmask field values.
    """
    @classmethod
    def pack(cls, **kwargs):
        """Pack a bitmask from explicit bit values.

        Parameters
        ----------
        kwargs
            The names of the fields and their status. Any fields not explicitly
            passed will be set to False.

        Returns
        -------
        bitmask : int
            The packed bitmask.
        """
        members = cls.__members__
        try:
            return reduce(
                op.or_,
                (members[k] * bool(v) for k, v in kwargs.items()),
            )
        except KeyError as e:
            raise TypeError(f'{e} is not a member of {cls.__qualname__}')

    @classmethod
    def unpack(cls, bitmask):
        """Unpack a bitmask into a dictionary from field name to field state.

        Parameters
        ----------
        bitmask : int
            The bitmask to unpack.

        Returns
        -------
        status : dict[str, bool]
            The mapping from field name to field status.
        """
        return {k: bool(bitmask & v) for k, v in cls.__members__.items()}
