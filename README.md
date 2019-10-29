# Booster

A lightweight library to easy the training and the debugging of deep neural networks with PyTorch.

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

diagnostic = Diagnostic(data)
```

### Aggregator

A module to compute the running average of the diagnostics.

```python
from booster.data import Aggregator, Diagnostic

aggregator = Aggregator()
...
aggregator.initialize()
for x in loader:
  data = op_step(model, data)
  aggregator.update(data)

summmary = aggregator.data # summary is a Diagnostic
summmary = summary.to('cpu')
```

The output is a Diagnostic object and can easily be dumped to Tensorboard.

```python
# log to tensorboard
writer = SummaryWriter(log_dir="....")
summary.log(writer, global_step)
```

## Pipeline: model + evaluator

An Evaluator computes the loss and the diagnostics. The pipeline fuses the model forward pass with the evaluator and can be wrapped into a custom Dataparallel class that handles the diagnostics.

```python
# fuse model + evaluator
pipeline = BoosterPipeline(model, evaluator)

# wrap as DataParallel
parallel_pipeline = DataParallelPipeline(pipeline, device_ids=device_ids)

# evaluate model on multiple devices and gather loss and diagnostics
data = next(iter(loader))
loss, diagnostics = parallel_pipeline(data) 
```



