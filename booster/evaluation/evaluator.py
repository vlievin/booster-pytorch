from typing import *

from torch import nn, Tensor
from ..datastruct import Diagnostic

class Evaluator():
    def __call__(self, model: nn.Module, data: Tuple, **kwargs: Any) -> Tuple[Tensor, Diagnostic]:
        """
        Computes a forward pass of the model given data, compute the loss and returns diagnostics.

        Basic implementation:
        ```
        logits = model(data)
        loss = loss_fn(logits, data)
        diagnostics = Diagnostic( { 'data1' : {....}, 'data2': {....}} )
        return loss, diagnostics
        ```

        :param model: pytorch model
        :param data: input data
        :param kwargs: other keywords arguments
        :return: (loss, diagnostics)
        """

        raise NotImplementedError
