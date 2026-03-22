__all__ = ("midi2cps",)


def midi2cps(note):
    return 440.0 * 2 ** ((note - 69) / 12)
