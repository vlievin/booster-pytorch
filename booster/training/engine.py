import logging
import os
from enum import Enum
from functools import partial
from typing import *

import torch
from booster.datastruct import Aggregator, Diagnostic
from booster.logging import LoggerManager, BestScore
from booster.pipeline import Pipeline
from booster.utils.schedule import Schedule
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

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


class TriggerType(Enum):
    STEP = 'step'
    EPOCH = "epoch"


class Trigger():
    def __init__(self, trigger_type: TriggerType, frequency: int, action: Callable):
        self.trigger_type = trigger_type
        self.frequency = frequency
        self.action = action

    def __call__(self, step, epoch, *args, **kwargs):
        iteration = {TriggerType.STEP: step, TriggerType.EPOCH: epoch}[self.trigger_type]

        if (iteration + 1) % self.frequency == 0:
            return self.action(*args, **kwargs)


def update_lr(optimizer: torch.optim.Optimizer, lrs: Union[float, list]):
    """
    update the learning rate
    args:
        optimizer (torch.optim.Optimizer): PyTorch Optimizer
        lr: lr value
    """
    if isinstance(lrs, float):
        lrs = [lrs]

    assert len(lrs) == len(optimizer.param_groups)

    for param_group, lr in zip(optimizer.param_groups, lrs):
        param_group['lr'] = lr

class ParametersScheduler(object):
    def __init__(self, parameters: Dict, rules: Dict[str, Schedule], optimizer: torch.optim.Optimizer):
        self.parameters = parameters
        self.rules = rules
        self.optimizer = optimizer

    def update(self, step, epoch):
        for k, rule in self.rules.items():
            self.parameters[k] = rule(step, epoch, self.parameters[k])

            if k == 'lr':
                update_lr(self.optimizer, self.parameters['lr'])


def _to_device(x: Any, device: torch.device):
    if isinstance(x, Tensor):
        return x.to(device)
    else:
        return x


class Engine():
    """
    Functionalities:
    1. training
    2. validation
    3. schedule params
    4. track best score
    5. sample
    6. save and load model + states
    7. run test epoch
    """

    def __init__(self, training_task: Training,
                 validation_tasks: List[BaseTask],
                 test_task: Validation,
                 parameters_scheduler: ParametersScheduler,
                 epochs: int,
                 device: torch.device,
                 logdir: str = 'runs/',
                 key2track: Callable = lambda diagnostics: diagnostics['loss']['elbo'],
                 task2track: Optional[str] = None,
                 debugging=False,
                 samplers=[],
                 **kwargs):

        self.setup_logging(logdir)

        self.training_task = training_task
        self.validation_tasks = validation_tasks
        self.test_task = test_task
        self.parameters_manager = parameters_scheduler
        self.epochs = epochs
        self.logdir = logdir
        self.device = device
        self.to_device = partial(_to_device, device=device)

        # key to track to save the best model
        self.key2track = key2track
        # if not provided, take the first validation task
        if task2track is None:
            task2track = self.validation_tasks[0].key
        self.task2track = task2track

        # state
        self.global_step = 0
        self.epoch = 0
        self.best_validation_score = BestScore(step=0, epoch=0, value=-1e12)

        # logging
        self.logger = LoggerManager(logdir, **kwargs)

        # debugging mode
        self.debugging = debugging

        # samplers
        self.samplers = samplers

    def setup_logging(self, logdir):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                            datefmt='%m-%d %H:%M',
                            handlers=[logging.FileHandler(os.path.join(logdir, 'run.log')),
                                      logging.StreamHandler()])

    def train(self):

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch

            # training loop
            self.training_task.initialize()
            for data in tqdm(self.training_task.dataloader, desc=f'{self.training_task.key} Epoch {epoch}'):
                data = map(self.to_device, data)
                self.training_task.step(self.global_step, data, **self.parameters_manager.parameters)
                self.global_step += 1

                # update parameters
                self.parameters_manager.update(self.global_step, self.epoch)

                # exit
                if self.debugging:
                    break

            # log training summary
            self.log(self.training_task.key, self.training_task.summary)

            # evaluation tasks
            for task in self.validation_tasks:
                diagnostic = self.run_task(self.global_step, self.epoch, task, **self.parameters_manager.parameters)

                best_score = None
                if task.key == self.task2track:
                    # save best model
                    self.update_validation_score_and_save(task.pipeline, diagnostic)
                    best_score = self.best_validation_score

                # log validation
                self.log(task.key, diagnostic, best_score=best_score)

            # sample models
            for sampler in self.samplers:
                sampler.sample(self.global_step)

        # final test
        # TODO: free memory
        self.test_task.load_model()  # implement this
        self.run_task(self.best_validation_score.step, self.best_validation_score.step.epoch, self.test_task,
                      **self.parameters_manager.parameters)

    def log(self, key, summary, **kwargs):
        self.logger.log(key, self.global_step, self.epoch, summary, **kwargs)

    def update_validation_score_and_save(self, pipeline: Pipeline, diagnostic: Diagnostic):
        score = self.key2track(diagnostic)
        prev_score = self.best_validation_score.value
        if score > prev_score:
            self.best_validation_score = BestScore(step=self.global_step, epoch=self.epoch, value=score)

            path = os.path.join(self.logdir, "pipeline.pt")
            torch.save(pipeline.state_dict(), path)

    def run_task(self, global_step, epoch, task, **parameters) -> Diagnostic:
        task.initialize()
        for data in tqdm(task.dataloader, desc=f'{task.key} Epoch {epoch}'):
            data = map(self.to_device, data)
            task.step(global_step, data, **parameters)
            if self.debugging:
                break
        return task.summary
