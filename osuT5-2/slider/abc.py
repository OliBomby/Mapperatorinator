import abc

# re-export
abstractmethod = abc.abstractmethod


class ABCMeta(abc.ABCMeta):
    def __new__(cls, name, bases, dict_):
        self = super().__new__(cls, name, bases, dict_)

        # copy docstrings down to implementation if the method does not
        # explicitly overwrite it
        super_dispatch = super(self, self)
        for meth in self.__abstractmethods__:
            impl = getattr(self, meth)
            if impl.__doc__ is None:
                try:
                    impl.__doc__ = getattr(super_dispatch, meth).__doc__
                except AttributeError:
                    pass

        return self
