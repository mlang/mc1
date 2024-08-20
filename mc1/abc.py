import abc


class ArgExpander(metaclass=abc.ABCMeta):
    """An abstract base class to expand list arguments to tuples of instances."""

    def __new__(cls, *args, **kwargs):
        seq = (list, range, tuple)
        lengths = [len(arg) for arg in args if isinstance(arg, seq)]
        lengths.extend(len(v) for k, v in kwargs.items() if isinstance(v, seq))
        if not lengths:
            return super().__new__(cls)

        def item(arg, index: int):
            return arg[index % len(arg)] if isinstance(arg, seq) else arg

        return tuple(
            cls(
                *(item(arg, index) for arg in args),
                **{k: item(v, index) for k, v in kwargs.items()}
            ) for index in range(max(lengths))
        )
