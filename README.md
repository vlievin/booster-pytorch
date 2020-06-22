# Booster

A lightweight library to ease the training and tracking of deep neural networks with PyTorch. Data structures and paradigms.

## Installation

```
pip install git+https://github.com/vlievin/booster-pytorch.git
```

## Data Structures

### Diagnostic

A two level dictionary structure to store the model diagnostics. Compatible with Tensorboard datastructure.

Example:

```python
from booster import Diagnostic

data = {
'loss' : {'nll' : torch.tensor([45., 58.]), 'kl': torch.tensor([22., 18.])},
'info' : {'batch_size' : 2, 'runtime' : 0.01}
}

diagnostic = Diagnostic(data)
```

### Aggregator

A module to compute the running average of the diagnostics.

```python
from booster import Aggregator, Diagnostic

aggregator = Aggregator()
...
aggregator.initialize()
for x in data_loader:
  data = optimization_step(model, data)
  aggregator.update(data)

summmary = aggregator.data # summary is an instance of Diagnostic
summmary = summary.to('cpu')
```

The output is a Diagnostic object and can easily be logged to Tensorboard.

```python
from torch.utils.tensorboard import SummaryWriter
# log to tensorboard
writer = SummaryWriter(log_dir="...")
summary.log(writer, global_step)

```

## Evaluator

The Evaluator class is a template that aims to abstract advanced loss function and diagnostics computation into a single operator. The output `__call__()` method is tuple `(loss: torch.Tensor, diagnostics: Diagnostic, output: dict)`. In this package are provided a simple Evaluator for classifiers and a class for Variational Inference, compatible with the training of VAEs. Using the Evaluator aims at writing minimal code, for instance, computing the loss and logging the diagnostics for a simple classifier becomes: 

```python
from booster.evaluation import Classification
model = Classifier()
evaluator = Classification(categories=10)

# evaluate model
data = next(iter(loader))
loss, diagnostics, output = evaluator(model, data)

# log the diagnostics
diagnostics.log(writer, global_step)

```

## Pipeline: model + evaluator
 
The pipeline combines the model and the evaluator into a single `torch.nn.Module`. It comes with a custom `torch.nn.DataParallel` module that handles the Diagnostic datastructure, so any model with an arbitrarily complex loss function/diagnostics computation can be parallelized across multiple GPUs.

```python
from booster import Pipeline, DataParallelPipeline

# fuse model + evaluator
pipeline = Pipeline(model, evaluator)

# wrap as DataParallel
parallel_pipeline = DataParallelPipeline(pipeline, device_ids=device_ids)

# evaluate model on multiple devices and gather loss and diagnostics
data = next(iter(loader))
loss, diagnostics, output = parallel_pipeline(data) 
```



