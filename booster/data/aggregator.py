from numbers import Number
from typing import *

import torch
from torch import Tensor

from .diagnostic import Diagnostic


class Aggregator():
    """
    A data structure to store and summarize diagnostics data.
    Data is summarized using moving average across the first dimension (batch dimension).
    The data is kept on device.
    """

    def __init__(self):
        self.initialize()

    def initialize(self):
        """initialize aggregator"""
        self._data = Diagnostic()
        self._count = Diagnostic()

    def update(self, new_data: Diagnostic):
        """moving average update of the leaves : s_n+k = s_n + 1/n+k (x_n+k - k * s_n)"""
        for k, v in new_data.items():
            for k_, v_ in v.items():
                if v_ is not None:

                    print(k, k_, v_)

                    sum, count = self._sum_and_count(v_)

                    if k_ not in self._data[k].keys():
                        self._count[k][k_] = count
                        self._data[k][k_] = sum / count
                    else:
                        self._count[k][k_] += count
                        self._data[k][k_] += (sum - count * self._data[k][k_]) / self._count[k][k_]

    @staticmethod
    def _sum_and_count(v: Union[Number, Tensor]) -> Tuple[Union[Number, Tensor], Number]:
        """compute sum and count"""
        if isinstance(v, Tensor):
            if isinstance(v, torch.LongTensor):
                v = v.float()
            count = v.size(0) if v.dim() > 0 else 1
            sum = torch.sum(v, 0)
            count = count * torch.ones_like(sum)
            return sum, count
        else:
            return v, 1

    @property
    def data(self) -> Diagnostic:
        """
        return the data.
        :return: summary a dict of dict `with leaf equals to the mean of the series
        """
        summary = self._data
        if len(summary):
            # add the count to the summary
            N_iters = next(iter(next(iter(self._count.values())).values()))
            summary['info']['iters'] = N_iters

        return summary
