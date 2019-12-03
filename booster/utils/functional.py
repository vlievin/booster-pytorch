import collections
import operator
from functools import reduce
from typing import *

import torch
from torch import Tensor


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
