"""aggregate two logs `data1` and `data2` and log to tensorboard where `data_i` contains tensors"""

import sys;

sys.path.append("../../")

from booster.datastruct import Aggregator, Diagnostic
from torch.utils.tensorboard import SummaryWriter
from booster.utils import logging_sep
from torch import tensor

# create data of shape [batch_size, *dims], batch_size = 2, dims = (2,)
data1 = {'loss': {'nll': tensor([0.1, 0.2]), 'kl': tensor([[0.1, 1.1], [0.2, 1.2]])}, 'other': {'ex1': 0.3}}
data2 = {'loss': {'nll': tensor([0.3, 0.4]), 'kl': tensor([[0.3, 1.3], [0.4, 1.4]])}, 'other': {'ex1': 0.5}}

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
print(logging_sep("="))
print(summary)
print(logging_sep())
print("nll =", summary['loss']['nll'].data)
print(logging_sep())
print("kl =", summary['loss']['kl'].data)
print(logging_sep())
print("other.ex1 = ", summary['other']['ex1'].data)
print(logging_sep("="))

# log to tensorboard
writer = SummaryWriter(log_dir="../../tensorboard")
summary.log(writer, 1)
