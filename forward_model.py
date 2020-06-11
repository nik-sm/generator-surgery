"""
For each image recovery task, we define an explicit forward model.

NOTE - for most forward models, we treat each number in the image
as a "pixel", so it has H x W x Ch total pixels.
"""
import math
import sys
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DEVICE = 'cuda:0'


class ForwardModel(ABC):
    viewable = False

    @abstractmethod
    def __call__(self, img):
        pass

    @abstractmethod
    def __str__(self):
        pass


class NoOp(ForwardModel):
    viewable = True

    def __init__(self, **fm_kwargs):
        pass

    def __call__(self, img):
        return img

    def __str__(self):
        return 'NoOp'


def get_random_mask(img_shape: Tuple[int, int, int],
                    fraction_kept: float = None,
                    n_kept: int = None,
                    device=None):
    """
    For image of shape CHW, returns random boolean
    mask of the same shape.
    """
    if n_kept is None and fraction_kept is None:
        raise ValueError()

    n_pixels = np.prod(img_shape)
    if fraction_kept:
        n_kept = int(fraction_kept * n_pixels)

    mask = torch.zeros(img_shape)
    if device:
        mask = mask.to(device)

    random_coords = torch.randperm(int(n_pixels))
    for i in range(n_kept):
        random_coord = np.unravel_index(random_coords[i], img_shape)
        mask[random_coord] = 1
    return mask


class GaussianCompressiveSensing(ForwardModel):
    """
    Wide Gaussian measurement matrix
    img_shape - 3 x H x W
    n_measure - number of rows for the measurement matrix
    """
    def __init__(self, img_shape, n_measure, device=DEFAULT_DEVICE):
        self.n_measure = n_measure
        self.img_shape = img_shape
        self.device = device
        self.A = torch.randn(np.prod(img_shape), n_measure, device=device)
        self.A /= math.sqrt(n_measure)

    def __call__(self, img):
        return img.view(img.shape[0], -1) @ self.A

    def __str__(self):
        return f'GaussianCompressiveSensing.n_measure={self.n_measure}'


class InpaintingScatter(ForwardModel):
    """
    Mask random pixels
    """
    viewable = True

    def __init__(self, img_shape, fraction_kept, device=DEFAULT_DEVICE):
        """
        img_shape - 3 x H x W
        fraction_kept - number in [0, 1], what portion of pixels to retain
        """
        assert fraction_kept <= 1 and fraction_kept >= 0
        self.fraction_kept = fraction_kept
        self.A = get_random_mask(img_shape,
                                 fraction_kept=self.fraction_kept).to(device)

    def __call__(self, img):
        return self.A[None, ...] * img

    def __str__(self):
        return f'InpaintingScatter.fraction_kept={self.fraction_kept}'


class SuperResolution(ForwardModel):
    viewable = True

    def __init__(self,
                 scale_factor,
                 mode='linear',
                 align_corners=True,
                 **kwargs):
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img):
        res = F.interpolate(img,
                            scale_factor=self.scale_factor,
                            mode=self.mode,
                            align_corners=self.align_corners)
        return res

    def __str__(self):
        return (f'SuperResolution.scale_factor={self.scale_factor}'
                f'.mode={self.mode}')


def get_forward_model(fm_name, **fm_kwargs):
    return getattr(sys.modules[__name__], fm_name)(**fm_kwargs)
