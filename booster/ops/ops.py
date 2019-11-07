from time import time
from typing import *

import torch

from ..data import Diagnostic
from ..pipeline import BoosterPipeline


def append_ellapsed_time(func):
    """append the elapsed time to the diagnostics"""

    def wrapper(*args, **kwargs):
        start_time = time()
        diagnostics = func(*args, **kwargs)
        diagnostics['info']['elapsed-time'] = time() - start_time
        return diagnostics

    return wrapper


@append_ellapsed_time
def training_step(pipeline: BoosterPipeline, data: Tuple, optimizer: torch.optim.Optimizer, iteration: int,
                  gradient_accumulation_steps: int = 1, max_grad_norm: Optional[float] = None,
                  **kwargs: Any) -> Diagnostic:
    """
    Perform a training step given a batch of data for a [model+evaluator] pipeline.
    Also performs:
    * Exponential Moving Average (EMA)
    * gradients clipping (max_grad_norm)
    * gradients accumulation (gradient_accumulation_steps)

    :param pipeline: model + evaluator
    :param data: batch of data
    :param optimizer: pytorch optimizer
    :param iteration: global step value
    :param gradient_accumulation_steps: number of steps used to accumulate gradients
    :param max_grad_norm: maximum norm of the gradients (None no clipping is applied)
    :param kwargs: additional args passed to the pipeline
    :return: diagnostics from the pipeline
    """
    pipeline.model.train()

    # process data using model and evaluator
    loss, diagnostics = pipeline(data, **kwargs)
    loss = loss.mean(0)

    # abort if loss is nan
    if loss != loss:
        raise ValueError(f"NaN encountered in loss computation at step {iteration}.")

    # loss scaling
    if gradient_accumulation_steps > 1:
        loss = loss / float(gradient_accumulation_steps)

    # compute backward pass
    loss.backward()

    # gradient clipping
    if max_grad_norm is not None and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), max_grad_norm)

    # optimizer and lr scheduler
    if (iteration + 1) % gradient_accumulation_steps == 0:
        # perform one optimization step
        optimizer.step()

        # re-initialize optimizer
        optimizer.zero_grad()

    return diagnostics


@torch.no_grad()
@append_ellapsed_time
def validation_step(pipeline: BoosterPipeline, data: Tuple, **kwargs: Any) -> Diagnostic:
    """
    Perform a validation step given a batch of data for a [model+evaluator] pipeline.

    :param pipeline: model + evaluator
    :param data: batch of data
    :param kwargs: additional args passed to the pipeline
    :return: diagnostics from the pipeline
    """
    pipeline.model.eval()
    _, diagnostics = pipeline(data, **kwargs)
    return diagnostics
