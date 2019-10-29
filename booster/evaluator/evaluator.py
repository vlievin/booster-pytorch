from typing import *

from torch import nn, Tensor
from ..data import Diagnostic

class Evaluator():
    def __call__(self, model: nn.Module, data: Tuple, **kwargs: Any) -> Tuple[Tensor, Diagnostic]:
        """

        :param model: pytorch model
        :param data: input data
        :param kwargs: other keywords arguments
        :return: (loss, diagnostics)
        """

        raise NotImplementedError
