from collections import defaultdict
from functools import partial
from typing import *

from torch import Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..utils import detach_to_device, detach


class Diagnostic(defaultdict):
    """
    A data structure to store the model's evaluation details.
    This is a two levels dictionary where each leaf is a value to track.
    Diagnostics are designed to log only numerical values.
    The structure matches with Tensorboard data logging specifications.

    Example:
    ```
    diagnostics = {
    'loss' : {'nll' : tensor, 'kl': tensor},
    'info : {'batch_size' : integer, 'runtime' : float}
    }
    ```

    """

    def __init__(self, __m: Optional[Mapping] = None):
        super().__init__(lambda: defaultdict(lambda: 0.))
        if __m is not None:
            self.update(__m)

    def __repr__(self):
        s = f"Diagnostic ({len(self)} keys):"
        if len(self):
            for k, v in self.items():
                s += f"\n\t{k} = ["
                for k_, v_ in v.items():
                    if isinstance(v_, (Tensor, np.ndarray)):
                        s += f"{k_} : {v_.shape}, "
                    else:
                        s += f"{k_} : {v_}, "
                s += "]"
        return s

    def update(self, __m: Mapping, **kwargs):
        for k, v in __m.items():
            if isinstance(v, Mapping):
                self[k] = Diagnostic.update(self.get(k, {}), v)
            else:
                self[k] = detach(v)

        return self

    def to(self, device):
        self.device = device
        format = partial(detach_to_device, device=device)
        return Diagnostic.map_nested_dicts(self, format)

    @staticmethod
    def map_nested_dicts(ob, func):
        if isinstance(ob, Mapping):
            data = {k: Diagnostic.map_nested_dicts(v, func) for k, v in ob.items()}
            return Diagnostic(data)
        else:
            return func(ob)

    def log(self, writer: SummaryWriter, global_step: int):
        """log Diagnostic to tensorboard"""
        for k, v in self.items():
            for k_, v_ in v.items():
                writer.add_scalar(str(k) + '/' + str(k_), v_, global_step)
