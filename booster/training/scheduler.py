import pickle
from typing import *

import torch

from booster.utils import Schedule


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

    def state_dict(self):
        return self.parameters

    def load_state_dict(self, data):
        self.parameters = data