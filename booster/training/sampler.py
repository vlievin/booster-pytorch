import torch
from booster.utils import make_grid_from_images

from .engine import BaseTask
from ..logging import LoggerManager


class Sampler():

    def sample(self, step: int, epoch: int, logger: LoggerManager):
        raise NotImplementedError


class PriorSampler(Sampler):
    def __init__(self, task: BaseTask, n_samples: int, sample=True, params=dict(), seed=None):
        self.key = task.key
        self.model = task.pipeline.model
        self.likelihood = task.pipeline.evaluator.likelihood
        self.n_samples = n_samples
        self.sample_likelihood = sample
        self.params = params
        self.seed = seed

    @torch.no_grad()
    def sample(self, step: int, epoch: int, logger: LoggerManager):
        self.model.eval()

        if self.seed is not None:
            torch.manual_seed(self.seed)

        print(f"# Sampling with seed = {self.seed}, params = {self.params}")

        samples = self.model.sample_from_prior(self.n_samples, **self.params)['x_']
        if self.sample_likelihood:
            samples = self.likelihood(logits=samples).sample()
        grid = make_grid_from_images(samples)

        logger.log_image(self.key, "prior_sample", step, epoch, grid)
