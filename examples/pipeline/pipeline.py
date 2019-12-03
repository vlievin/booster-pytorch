import sys;

sys.path.append("../../")

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from booster.evaluation import Classification
from booster.pipeline import Pipeline, DataParallelPipeline

# load data
dataset = torchvision.datasets.MNIST('../../data', train=False, download=True,
                                     transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# define model
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        print(f"Classifier.forward: x.device = {x.device}, x.shape = {x.shape}")

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# check GPUs
n_gpus = torch.cuda.device_count()
device_ids = list(range(n_gpus)) if n_gpus else None
print("N gpus:", n_gpus, "Devices:", device_ids)
assert n_gpus > 1

# init model and evaluator
model = Classifier()
evaluator = Classification(10)
model.to("cuda:0")

# fuse model + evaluator
pipeline = Pipeline(model, evaluator)

# wrap as DataParallel
parallel_pipeline = DataParallelPipeline(pipeline, device_ids=device_ids)

# evaluate model
data = next(iter(loader))
print("x.shape =", next(iter(data)).shape)
loss, diagnostics = parallel_pipeline(data)

print("\nLoss:", loss.device, loss)
print(diagnostics)
