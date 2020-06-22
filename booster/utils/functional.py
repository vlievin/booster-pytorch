import collections
import math
import operator
from functools import reduce
from typing import *

import torch
from torch import Tensor
from torchvision.utils import make_grid


def prod(x: Iterable):
    """return the product of an Iterable"""
    if len(x):
        return reduce(operator.mul, x)
    else:
        return 0


def batch_reduce(x: Tensor, reduce=torch.sum):
    """reduce each batch element of a tensor"""
    batch_size = x.size(0)
    return reduce(x.view(batch_size, -1), dim=-1)


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum, eps: float = 1e-12, keepdim=False):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=keepdim)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=keepdim) + eps) + max


def detach(x):
    """detach, clone and or place on the given device"""
    if x is not None:
        if isinstance(x, torch.Tensor):
            return x.detach()
        else:
            return x
    else:
        return None

def detach_to_device(x, device):
    """detach, clone and or place on the given device"""
    if x is not None:
        if isinstance(x, torch.Tensor):
            return x.detach().clone().to(device)
        else:
            return torch.tensor(x, device=device, dtype=torch.float)
    else:
        return None


def safe_sum(x):
    """sum object if it is iterable else return value"""
    if isinstance(x, collections.abc.Iterable):
        return sum(x)
    else:
        return x


def _to_device(x: Any, device: torch.device):
    if isinstance(x, Tensor):
        return x.to(device)
    else:
        return x


def make_grid_from_images(x_: Tensor, nrow: Optional[int] = None) -> Tensor:
    print("# make_grid:", x_.min(), x_.max())

    # replace nans with 0
    x_[x_ != x_] = 0

    # normalization
    x_ = x_.float()

    # round minimum to lower integer
    _min = x_.min().long().float()
    if _min < 0 and _min > x_.min():
        _min -= 1

    x_ -= _min

    # round maximum to upper integer
    _max = x_.max().long().float()
    if _max > 0 and _max < x_.max():
        _max += 1

    x_ /= _max

    # make grid
    if nrow is None:
        nrow = math.floor(math.sqrt(x_.size(0)))
    return make_grid(x_, nrow=nrow)
