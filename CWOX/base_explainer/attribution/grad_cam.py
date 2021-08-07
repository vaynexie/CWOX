
__all__ = ["grad_cam"]

import torch
from .common import saliency


def gradient_to_grad_cam_saliency(x):
    r"""Convert activation and gradient to a Grad-CAM saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the saliency map :math:`s`: given by:

    .. math::

        s_{n1u} = \max\{0, \sum_{c}x_{ncu}\cdot dx_{ncu}\}

    Args:
        x (:class:`torch.Tensor`): activation tensor with a valid gradient.

    Returns:
        :class:`torch.Tensor`: saliency map.
    """
    # Apply global average pooling (GAP) to gradient.
    grad_weight = torch.mean(x.grad, (2, 3), keepdim=True)

    # Linearly combine activations and GAP gradient weights.
    saliency_map = torch.sum(x * grad_weight, 1, keepdim=True)

    # Apply ReLU to visualization.
    #saliency_map = torch.clamp(saliency_map, min=0)

    return saliency_map


def grad_cam(*args,
             saliency_layer,
             gradient_to_saliency=gradient_to_grad_cam_saliency,
             **kwargs):
    r"""Grad-CAM method.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the Grad-CAM method, and supports the
    same arguments and return values.
    """
    return saliency(*args,
                    saliency_layer=saliency_layer,
                    gradient_to_saliency=gradient_to_saliency,
                    **kwargs,)
