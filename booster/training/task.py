import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from booster.datastruct import Aggregator, Diagnostic
from booster.pipeline import Pipeline
from booster.utils.functional import _to_device
from .ops import training_step, validation_step


class BaseTask():
    def __init__(self,
                 key: str,
                 pipeline: Pipeline,
                 dataloader: DataLoader):
        super().__init__()

        self.key = key
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.aggregator = Aggregator()

    def initialize(self):
        self.aggregator.initialize()

    @property
    def summary(self) -> Diagnostic:
        return self.aggregator.data.to('cpu')

    def step(self, iteration, data, **kwargs):
        raise NotImplementedError

    def run_epoch(self, global_step):
        for data in tqdm(self.dataloader, desc=f'{self.key} Epoch'):
            data = map(_to_device, data)
            self.step(global_step, data, **self.parameters_manager)


class Training(BaseTask):
    def __init__(self,
                 key: str,
                 pipeline: Pipeline,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 **kwargs):
        super().__init__(key, pipeline, dataloader)

        self.optimizer = optimizer
        self.kwargs = kwargs

    def step(self, iteration, data, **kwargs):
        diagnostics = training_step(self.pipeline, data, self.optimizer, iteration, **self.kwargs, **kwargs)
        self.aggregator.update(diagnostics)


class Validation(BaseTask):
    def __init__(self,
                 key: str,
                 pipeline: Pipeline,
                 dataloader: DataLoader,
                 **kwargs):
        super().__init__(key, pipeline, dataloader)

        self.kwargs = kwargs

    def step(self, iteration, data, **kwargs):
        diagnostics = validation_step(self.pipeline, data, **self.kwargs, **kwargs)
        self.aggregator.update(diagnostics)
