from enum import Enum
from typing import *


class TriggerType(Enum):
    STEP = "step"
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
