# Booster

A lightweight library to ease the training and the debugging of deep neural networks with PyTorch. Data structures and paradigms.

## Data Structures

### Diagnostic

A two level dictionary structure to store the model diagnostics. Compatible with Tensorboard datastructure.

Example:

```python
from booster import Diagnostic

data = {
'loss' : {'nll' : [45., 58.], 'kl': [22., 18.]},
'info' : {'batch_size' : 16, 'runtime' : 0.01}
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

The base Evaluator class is a framework to perform loss and diagnostics computation. 

```python
from booster.evaluation import Classification
model = Classifier()
evaluator = Classification(categories=10)

# evaluate model
data = next(iter(loader))
loss, diagnostics, output = evaluator(model, data)

```

## Pipeline: model + evaluator
 
The pipeline fuses the model forward pass with the Evaluator and can be wrapped into a custom Dataparallel class that handles the diagnostics and outputs.The output is tuple (loss: torch.Tensor, diagnostics: Diagnostic, output: dict). Here is an example for a simple classifier. 

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



