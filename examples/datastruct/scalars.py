"""aggregate two logs `data1` and `data2` and log to tensorboard"""

import sys;

sys.path.append("../../")

from booster import Aggregator, Diagnostic
from torch.utils.tensorboard import SummaryWriter
from booster.utils import logging_sep

# create data
data1 = {'loss': {'nll': 0.1, 'kl': 0.5}, 'other': {'ex1': 0.3}}
data2 = {'loss': {'nll': 0.3, 'kl': 0.7}, 'other': {'ex1': 0.5}}

# create a diagnostic object for data1
diag1 = Diagnostic(data1)

# create a diagnostic object for data2
diag2 = Diagnostic()
diag2.update(data2)

# create an aggregator
agg = Aggregator()

# leaf average of diag1 and diag2
agg.update(diag1)
agg.update(diag2)

# return moving average and move to CPU
summary = agg.data.to('cpu')

# print resulting summary
print(logging_sep())
print(summary)
print(logging_sep())

# log to tensorboard
writer = SummaryWriter(log_dir="../../tensorboard")
summary.log(writer, 1)
