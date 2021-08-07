

__all__ = ["gradient"]

from .common import saliency


def gradient(*args, context_builder=None, **kwargs):
    r"""Gradient method

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the gradient method, and supports the
    same arguments and return values.
    """
    assert context_builder is None
    return saliency(*args, **kwargs)
