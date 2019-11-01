from typing import *

import torch
from torch import nn, Tensor

from .evaluator import Evaluator
from ..data import Diagnostic
from ..utils import batch_reduce


class Classification(Evaluator):

    def __init__(self, categories: int):
        self.categories = categories
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, model: nn.Module, data: Tensor, **kwargs: Any) -> Tuple[Tensor, Diagnostic]:
        """
        Compute the logits given the model and the data, compute the loss, and return loss + diagnostics

        :param model: pytorch model
        :param data: input data
        :param kwargs: other keywords arguments
        :return: (loss, diagnostics)
        """

        # unpack data
        x, y, *_ = data

        # compute forward pass
        y_ = model(x, **kwargs)

        # compute negative log likelihood
        nll = self.loss_fn(y_, y)
        nll = batch_reduce(nll)

        # compute accuracy
        matches = (y_.argmax(1) == y).float()
        batch_accuracy = matches.sum() / torch.ones_like(matches).sum()

        # diagnostics
        diagnostics = {
            'loss': {'nll': nll, "batch_accuracy": batch_accuracy},
            'info': {'batch_size': x.size(0)}
        }

        # create diagnostics object and convert everything into tensors on x.device
        diagnostics = Diagnostic(diagnostics).to(x.device)

        return nll.mean(0), diagnostics
