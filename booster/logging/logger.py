import logging
import os
import sys
import warnings
from collections import namedtuple
from typing import *

import matplotlib.image
from booster import Diagnostic
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

BestScore = namedtuple('BestScore', ['step', 'epoch', 'value', 'summary'])


class BaseLogger():
    def __init__(self, key, logdir):
        self.key = key
        self.logdir = logdir

    def log_diagnostic(self, global_step: int, epoch: int, summary: Diagnostic, **kwargs):
        raise NotImplementedError

    def log_image(self, key: str, global_step: int, epoch: int, img_tensor: Tensor):
        raise NotImplementedError


class TensorboardLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self.writer = SummaryWriter(os.path.join(self.logdir, self.key))

    def log_diagnostic(self, global_step: int, epoch: int, summary: Diagnostic, **kwargs):
        summary.log(self.writer, global_step)

    def log_image(self, key: str, global_step: int, epoch: int, img_tensor: Tensor):
        self.writer.add_image(key, img_tensor, global_step=global_step)


class LoggingLogger(BaseLogger):
    def __init__(self, *args, diagnostic_keys=['loss'], **kwargs):
        super().__init__(*args)

        self.logger = logging.getLogger(self.key)

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.logger = logging.getLogger(self.key)

        fileHandler = logging.FileHandler(os.path.join(self.logdir, 'run.log'))
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        self.logger.addHandler(consoleHandler)

        self.diagnostic_keys = diagnostic_keys

    def log_diagnostic(self, global_step: int, epoch: int, summary: Diagnostic, best_score: Optional[BestScore] = None,
                       **kwargs):
        for stats_key in self.diagnostic_keys:
            if not stats_key in summary.keys():
                self.logger.warning('key ' + str(stats_key) + ' not in summary.')
            else:
                message = f'[{global_step} / {epoch}]   '
                message += ''.join([f'{k} {v:6.2f}   ' for k, v in summary.get(stats_key).items()])
                if "info" in summary.keys() and "elapsed-time" in summary["info"].keys():
                    message += f'({summary["info"]["elapsed-time"]:.2f}s /iter)'
                else:
                    warnings.warn(
                        f"Summary does not contain the key info/elapsed-time. The elapsed time won't be displayed.")
                if best_score is not None:
                    message += f'   (best: {best_score.value:6.2f}  [{best_score.step} | {best_score.epoch}])'

            self.logger.info(message)

    def log_image(self, key: str, global_step: int, epoch: int, img_tensor: Tensor):
        img = img_tensor.data.permute(1, 2, 0).cpu().numpy()
        matplotlib.image.imsave(os.path.join(self.logdir, f"{key}.png"), img)


class Logger(BaseLogger):
    def __init__(self, key, logdir, tensorboard=True, logging=True, **kwargs):
        super().__init__(key, logdir)

        self.loggers = []

        if tensorboard:
            self.loggers += [TensorboardLogger(key, logdir, **kwargs)]

        if logging:
            self.loggers += [LoggingLogger(key, logdir, **kwargs)]

    def log_diagnostic(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_diagnostic(*args, **kwargs)

    def log_image(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_image(*args, **kwargs)


class LoggerManager():

    def __init__(self, logdir, **kwargs):
        self.logdir = logdir
        self.kwargs = kwargs

        self.loggers = {}

    def init_logger(self, key):
        self.loggers[key] = Logger(key, self.logdir, **self.kwargs)

    def log_diagnostic(self, key, step, epoch, summary, **kwargs):
        if key not in self.loggers:
            self.init_logger(key)

        self.loggers[key].log_diagnostic(step, epoch, summary, **kwargs)

    def log_image(self, key, image_key, step, epoch, img_tensor, **kwargs):
        if key not in self.loggers:
            self.init_logger(key)

        self.loggers[key].log_image(image_key, step, epoch, img_tensor, **kwargs)
