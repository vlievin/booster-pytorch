import argparse

from booster.evaluation import VariationalInference
from booster.models import SimpleVAE
from booster.training.engine import *
from booster.training.engine import ParametersScheduler
from booster.training.sampler import PriorSampler
from booster.utils.schedule import DecaySchedule, LinearSchedule
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Lambda, ToTensor, Compose

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='runs/', help='directory to store training logs')
parser.add_argument('--run_id', default='example', help='run identifier')
parser.add_argument('--data_root', default='data/', help='directory to store the dataset')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--lr', default=2e-3, type=float, help='base learning rate')
parser.add_argument('--seed', default=42, type=int, help='random seed')

opt = parser.parse_args()

# random seed
torch.manual_seed(opt.seed)

# logging
logdir = os.path.join(opt.root, opt.run_id)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# dataset
_transform = Compose([ToTensor(), Lambda(lambda x: (x > 0.5).float())])
train_dataset = MNIST('data/', train=True, transform=_transform)
test_dataset = MNIST('data/', train=False, transform=_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=opt.bs, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2 * opt.bs, shuffle=False, num_workers=0)

# sample
sample, *_ = next(iter(train_loader))
tensor_shape = (-1, *sample.shape[1:])

# model
model = SimpleVAE(tensor_shape, 32, 256, 2)

# likelihood
likelihood = Bernoulli

# evaluator
evaluator = VariationalInference(likelihood, iw_samples=1)
test_evaluator = VariationalInference(likelihood, iw_samples=100)

# pipeline
pipeline = Pipeline(model, evaluator)
test_pipeline = Pipeline(model, test_evaluator)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

# scheduling
parameters = {'beta': 1e-3, 'freebits': [2.0], 'lr': opt.lr}
rules = {
    'beta': LinearSchedule(1e-3, 1, 10000),
    'lr': DecaySchedule(0.999, 1e-4)
}
parameters_scheduler = ParametersScheduler(parameters, rules, optimizer)

# tasks
training_task = Training("Training", pipeline, train_loader, optimizer)
validation_task = Validation("Validation", pipeline, test_loader)
test_task = Validation("Test", test_pipeline, test_loader)

# sampler
prior_sampler = PriorSampler(validation_task, 100)
samplers = [prior_sampler]

# define engine
device = "cuda" if torch.cuda.is_available() else "cpu"
key2track = lambda diagnostic: diagnostic['loss']['elbo']
engine = Engine(training_task, [validation_task], test_task, parameters_scheduler, opt.epochs, device, logdir,
                key2track, samplers=samplers)

# training
engine.train()

# load best model
engine.load_best_model(pipeline)

# sample
engine.sample()
