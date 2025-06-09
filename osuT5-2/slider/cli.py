from contextlib import contextmanager


def maybe_show_progress(it, show_progress, **kwargs):
    """Optionally show a progress bar for the given iterator.

    Parameters
    ----------
    it : iterable
        The underlying iterator.
    show_progress : bool
        Should progress be shown.
    **kwargs
        Forwarded to the click progress bar.

    Returns
    -------
    itercontext : context manager
        A context manager whose enter is the actual iterator to use.

    Examples
    --------
    .. code-block:: python

       with maybe_show_progress([1, 2, 3], True) as ns:
            for n in ns:
                ...
    """
    if show_progress:
        try:
            import click
        except ImportError as e:
            raise ImportError(
                "click must be installed to show a progressbar"
            ) from e
        return click.progressbar(it, **kwargs)

    @contextmanager
    def ctx():
        yield it

    return ctx()
