import logging
import os
import sys
import warnings
from collections import namedtuple
from typing import *

import matplotlib.image
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from booster import Diagnostic
from .datatracker import DataTracker

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

        # logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        # self.logger = logging.getLogger(self.key)
        #
        # fileHandler = logging.FileHandler(os.path.join(self.logdir, 'run.log'))
        # fileHandler.setFormatter(logFormatter)
        # self.logger.addHandler(fileHandler)
        #
        # consoleHandler = logging.StreamHandler(sys.stdout)
        # consoleHandler.setFormatter(logFormatter)
        # self.logger.addHandler(consoleHandler)

        self.logger.setLevel(logging.INFO)

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
        pass


class PlotLogger(BaseLogger):
    def __init__(self, *args, diagnostic_keys=['loss'], **kwargs):
        super().__init__(*args)
        self.diagnostic_keys = diagnostic_keys
        self.tracker = DataTracker(label=self.key)

    def log_diagnostic(self, global_step: int, epoch: int, summary: Diagnostic, **kwargs):
        for key in self.diagnostic_keys:
            self.tracker.append(global_step, summary[key])

    def plot(self, *args, **kwargs):
        self.tracker.plot(*args, **kwargs)

    def log_image(self, key: str, global_step: int, epoch: int, img_tensor: Tensor):
        img = img_tensor.data.permute(1, 2, 0).cpu().numpy()
        matplotlib.image.imsave(os.path.join(self.logdir, f"{key}.png"), img)


class PlotHandler(List):
    def __init__(self, logdir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = os.path.join(logdir, "curves.png")

    def plot(self):

        if len(self):
            logger = self[0]
            keys = logger.tracker.data.keys()

            plt.figure(figsize=(4 * len(keys), 3))
            for i, key in enumerate(keys):
                plt.subplot(1, len(keys), i + 1)
                plt.title(key)
                for logger in self:
                    logger.plot(key)
                plt.legend()

            plt.savefig(self.path)


class Logger(BaseLogger):
    def __init__(self, key, logdir, tensorboard=True, logging=True, plot=True, **kwargs):
        super().__init__(key, logdir)

        self.loggers = []

        if tensorboard:
            self.loggers += [TensorboardLogger(key, logdir, **kwargs)]

        if logging:
            self.loggers += [LoggingLogger(key, logdir, **kwargs)]

        if plot:
            self.loggers += [PlotLogger(key, logdir, **kwargs)]

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

        self.plot_handler = PlotHandler(self.logdir)

    def init_logger(self, key):
        self.loggers[key] = Logger(key, self.logdir, **self.kwargs)

        # mappend PlotLogger to PlotHandler
        for logger in self.loggers[key].loggers:
            if isinstance(logger, PlotLogger):
                self.plot_handler.append(logger)

    def log_diagnostic(self, key, step, epoch, summary, **kwargs):
        if key not in self.loggers:
            self.init_logger(key)

        self.loggers[key].log_diagnostic(step, epoch, summary, **kwargs)

        self.plot_handler.plot()

    def log_image(self, key, image_key, step, epoch, img_tensor, **kwargs):
        if key not in self.loggers:
            self.init_logger(key)

        self.loggers[key].log_image(image_key, step, epoch, img_tensor, **kwargs)
