# Booster

A lightweight library to train deep neural networks with PyTorch.

## Data Structures

### Diagnostic

A two level dictionary structure to store the model diagnostics. Compatible with Tensorboard datastructure.

Example:

```python
from booster.data import Diagnostic

data = {
'loss' : {'nll' : [45., 58.], 'kl': [22., 18.]},
'info : {'batch_size' : 16, 'runtime' : 0.01}
}

diagnostics = Diagnostics(data).to(device)
```

### Aggregator

A module to compute the running average of the diagnostics.

```python
from booster.data import Aggregator, Diagnostic

data1 = Diagnostics({
'loss' : {'nll' : [45., 58.], 'kl': [22., 18.]},
'info : {'batch_size' : 16, 'runtime' : 0.01}
})

data2 = Diagnostics({
'loss' : {'nll' : [45., 58.], 'kl': [22., 18.]},
'info : {'batch_size' : 16, 'runtime' : 0.01}
})

aggregator = Aggregator()
aggregator.update(data1)
aggregator.update(data2)

summmary = aggregator.data # summary is a Diagnostic
summmary = summary.to('cpu')
```

The output is a Diagnostic object and can easily be dumped to Tensorboard.

```python
# log to tensorboard
writer = SummaryWriter(log_dir="....")
summary.log(writer, global_step)
```

## Pipeline: Model + Evaluator

An Evaluator computes the loss and the diagnostics. The pipeline fuses the model forward pass with the evaluator and can be wrapped into a custom Dataparallel class that handles the diagnostics.






