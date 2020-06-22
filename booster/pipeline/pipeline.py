from typing import *

import torch
from torch import nn, Tensor
from torch.nn.parallel.data_parallel import DataParallel

from ..datastruct import Diagnostic
from ..evaluation import Evaluator


class Pipeline(torch.nn.Module):
    """
    fuse model forward op with evaluation forward op to ease the DataParallel logic
    """

    def __init__(self, model: nn.Module, evaluator: Evaluator):
        super().__init__()
        self.model = model
        self.evaluator = evaluator

    def forward(self, data, **kwargs):
        return self.evaluator(self.model, data, **kwargs)


class DataParallelPipeline(DataParallel):
    """
    A DataParallel wrapper for BoosterPipelines: handles diagnostics.
    Take the average of the loss across GPUs
    """

    def forward(self, *inputs, **kwargs) -> Tuple[Tensor, Diagnostic, Dict]:
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        output_per_gpu = self.parallel_apply(replicas, inputs, kwargs)
        loss, diagnostics, output = self.unpack(output_per_gpu)

        # gather loss
        loss = self.gather(loss, self.output_device)

        # gather diagnostics
        diagnostics = self.gather_and_reduce_diagnostics(diagnostics, self.output_device)

        # gather output
        output = self.gather_output(output, self.output_device)

        return loss.mean(0), diagnostics, output

    @staticmethod
    def unpack(outputs):
        """ unpack tuple(gpu_0, gpu_1, ..) to (loss_0, loss_1, ..), (diagnostics_0, ..),  (output_0, ..)"""
        return zip(*outputs)

    def safe_gather(self, data, device):
        if len(data):
            d_first, *_ = data
            if isinstance(d_first, torch.Tensor):
                data = self.gather(data, device)
            else:
                raise ValueError(f"Received a diagnostics containing an instance of type{type(d_first)}, "
                                 f"which is not implemented.")

            return data
        else:
            return None

    def gather_output(self, outputs: List[Dict], device: torch.device) -> Dict:
        sample, *_ = outputs
        return {k : self.safe_gather([o[k] for o in outputs], device) for k in sample.keys()}

    def gather_and_reduce_diagnostics(self, diagnostics: Diagnostic, device: torch.device) -> Diagnostic:
        """gather op for diagnostics"""
        sample, *_ = diagnostics

        def gather_and_reduce_leaf(key_1, key_2, device):
            # get value from each device
            data = [d[key_1][key_2] for d in diagnostics]

            # filter nans
            data = [d for d in data if d is not None]

            return self.safe_gather(data, device)

        diagnostics = {k: {k_: gather_and_reduce_leaf(k, k_, device) for k_, v_ in v.items()} for k, v in
                       sample.items()}

        return Diagnostic(diagnostics)
