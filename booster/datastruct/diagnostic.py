from collections import defaultdict
from functools import partial
from typing import *

import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ..utils import detach_to_device, detach


class Diagnostic(defaultdict):
    """
    A data structure to store the model's evaluation details.
    This is a two levels dictionary where each leaf is a value being tracked.
    Leaf values can be either scalars or Tensors.

    The data structure matches with Tensorboard data logging specifications, so logging can be done seamlessly.
    Warning: For tensors with more than one dimension, the mean value will be logged to Tensorboard

    Example:
    ```
    tensor = torch.tensor([[0.1, 1.1], [0.2, 1.2]]) # shape [batch_size, *dims] = [2, 2]
    diagnostics = Diagnostic({
    'loss' : {'nll' : tensor, 'kl': tensor},
    'info' : {'batch_size' : 2, 'runtime' : 1.367}
    })
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
                s += f"\n  {k} : {{"
                for k_, v_ in v.items():
                    if isinstance(v_, Tensor):
                        s += f"\n    {k_} : mean = {v_.mean().item():.3f}, std = {v_.std().item():.3f}, shape = {tuple(v_.shape)}, dtype = {v_.dtype}, device = {v_.device},"
                    elif isinstance(v_, np.ndarray):
                        s += f"\n    {k_} : mean = {v_.mean():.3f}, std = {v_.std():.3f}, shape = {tuple(v_.shape)}, dtype = {v_.dtype},"
                    else:
                        s += f"\n    {k_} : value = {v_:.3f}, type = {type(v_)},"
                s += "\n  }"
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
        """log Diagnostic to tensorboard, if the leaf variable is a tensor, log the mean value"""
        for k, v in self.items():
            for k_, v_ in v.items():
                if isinstance(v_, Tensor) and v_.ndim > 0:
                    v_ = v_.mean()
                writer.add_scalar(str(k) + '/' + str(k_), v_, global_step)
