import json
import logging
import os
from collections import defaultdict
from functools import partial
from typing import *

import torch
from booster.datastruct import Aggregator, Diagnostic
from booster.logging import LoggerManager, BestScore
from booster.pipeline import Pipeline
from booster.utils.functional import _to_device
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ops import training_step, validation_step
from .scheduler import ParametersScheduler


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


class Engine():
    """

    TODO:
    * save state
    * loading / saving best model
    * run test epoch
    * overall UX

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

        print(f"# logging directory: {os.path.abspath(logdir)}")

        self.setup_logging(logdir)

        self.training_task = training_task
        self.validation_tasks = validation_tasks
        self.test_task = test_task
        self.parameters_manager = parameters_scheduler
        self.epochs = epochs
        self.logdir = logdir
        self.model_path = os.path.join(self.logdir, "pipeline.pt")
        self.eval_score_path = os.path.join(self.logdir, "eval-score.json")
        self.test_score_path = os.path.join(self.logdir, "test-score.json")
        self.device = device

        # key to track to save the best model
        self.key2track = key2track
        # if not provided, take the first validation task
        if task2track is None:
            task2track = self.validation_tasks[0].key
        self.task2track = task2track

        # state
        self.global_step = 0
        self.epoch = 0
        self.best_validation_score = BestScore(step=0, epoch=0, value=-1e12, summary=None)

        # logging
        self.logger = LoggerManager(logdir, **kwargs)

        # debugging mode
        self.debugging = debugging

        # samplers
        self.samplers = samplers

        M_parameters = (sum(p.numel() for p in self.training_task.pipeline.model.parameters() if p.requires_grad) / 1e6)
        self.info_logger.info(f'# Total Number of Parameters: {M_parameters:.3f}M')

    def setup_logging(self, logdir):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s',
                            datefmt='%m-%d %H:%M',
                            handlers=[logging.FileHandler(os.path.join(logdir, 'run.log')),
                                      logging.StreamHandler()])

        self.info_logger = logging.getLogger("Info")

    @staticmethod
    def to_device(data, device):
        if isinstance(data, Tensor):
            return data.to(device)
        else:
            to_device = partial(_to_device, device=device)
            return map(to_device, data)

    def train(self):

        # sample models
        self.sample()

        # evaluation tasks
        self.evaluate()

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch

            # training loop
            self.training_task.initialize()
            for data in tqdm(self.training_task.dataloader, desc=f'{self.training_task.key} Epoch {epoch}'):
                data = self.to_device(data, self.device)
                self.training_task.step(self.global_step, data, **self.parameters_manager.parameters)
                self.global_step += 1

                # update parameters
                self.parameters_manager.update(self.global_step, self.epoch)

                # exit
                if self.debugging:
                    break

            # log training summary
            self.log_diagnostic(self.training_task.key, self.training_task.summary)

            # evaluation tasks
            self.evaluate()

            # sample models
            self.sample()

    def load_best_model(self, pipeline):
        pipeline.load_state_dict(torch.load(self.model_path))

    def sample(self):
        for sampler in self.samplers:
            sampler.sample(self.global_step, self.epoch, self.logger)

    def log_diagnostic(self, key, summary, **kwargs):
        self.logger.log_diagnostic(key, self.global_step, self.epoch, summary, **kwargs)

    def update_validation_score_and_save(self, pipeline: Pipeline, diagnostic: Diagnostic):
        score = self.key2track(diagnostic)
        prev_score = self.best_validation_score.value
        if score > prev_score:
            self.best_validation_score = BestScore(step=self.global_step, epoch=self.epoch, value=score,
                                                   summary=diagnostic)

            # save metadata
            self.save_score(self.best_validation_score, self.eval_score_path)

            # save model
            torch.save(pipeline.state_dict(), self.model_path)

    def save_score(self, best_score, path):
        data = best_score._asdict()

        data['value'] = data['value'].mean().item()

        summary = defaultdict(dict)
        for k, v in data['summary'].items():
            for kk, vv in v.items():
                if isinstance(vv, Tensor):
                    vv = vv.mean().item()
                summary[k][kk] = vv
        data['summary'] = summary

        with open(path, 'w') as fp:
            json.dump(data, fp)

    def evaluate(self):
        for task in self.validation_tasks:
            diagnostic = self.run_task(self.global_step, self.epoch, task, **self.parameters_manager.parameters)

            best_score = None
            if task.key == self.task2track:
                # save best model
                self.update_validation_score_and_save(task.pipeline, diagnostic)
                best_score = self.best_validation_score

            # log validation
            self.log_diagnostic(task.key, diagnostic, best_score=best_score)

    def test(self):
        data = self.run_task(self.global_step, self.epoch, self.test_task, **self.parameters_manager.parameters)

        summary = defaultdict(dict)
        for k, v in data.items():
            for kk, vv in v.items():
                if isinstance(vv, Tensor):
                    vv = vv.mean().item()
                summary[k][kk] = vv

        data = {'summary': summary}

        with open(self.test_score_path, 'w') as fp:
            json.dump(data, fp)

    def run_task(self, global_step, epoch, task, **parameters) -> Diagnostic:
        task.initialize()
        for data in tqdm(task.dataloader, desc=f'{task.key} Epoch {epoch}'):
            data = self.to_device(data, self.device)
            task.step(global_step, data, **parameters)
            if self.debugging:
                break
        return task.summary
