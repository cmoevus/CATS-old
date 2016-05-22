"""Transform input into a slice."""


def slicify(s, default=slice(None)):
    """
    Transform the input into a slice.

    Acceptable input for "s":
        slice
        list/tuple
        int
    default is a slice that contains the
    """
    # Arrange the default values
    if isinstance(default, slice) == False:
        default = slicify(default)

    # Arrange the values
    if isinstance(s, slice):
        pass
    elif hasattr(s, '__iter__'):
        if len(s) == 1 or s[1] is None:
            s = (s[0], default.stop)
        if s[0] is None:
            s = (default.start, s[1])
        s = slice(*s)
    elif isinstance(s, int):
        s = slice(s, s + 1)
    else:
        s = slicify(default)
    if s.step is None:
        s = slice(s.start, s.stop, default.step)
    return s
