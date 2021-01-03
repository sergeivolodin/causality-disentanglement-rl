import inspect
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size_test = 512
batch_size_train = 512

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fclr = nn.Linear(50, 1)

    def common(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

    def lr(self, x):
        x = self.common(x)
        x = self.fclr(x)
        return x

    def forward(self, x):
        x = self.common(x)
        x = self.fc2(x)
        return F.log_softmax(x)

def caller_name():
    """https://stackoverflow.com/questions/2654113/how-to-get-the-callers-method-name-in-the-called-method"""
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    return calframe[1][3]

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

class Interaction():
    def __init__(self):
        self.internals = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)
        # self.alpha = nn.Parameter(torch.tensor(1e-3), requires_grad=True).to(self.device)
        self.optimizer_cls = torch.optim.SGD
        self.data = test_loader, train_loader
        # self.hparams_old = self.alpha
        self.data_train = enumerate(self.data[1])


    @property
    def internal(self):
        c_name = caller_name()
        if c_name not in self.internals:
            self.internals[c_name] = {}
        return self.internals[c_name]

    def inference_step(self):
        try:
            batch_idx, (example_data, example_targets) = next(self.data_train)
        except StopIteration:
            self.data_train = enumerate(self.data[1])
            batch_idx, (example_data, example_targets) = next(self.data_train)
        out = self.model(example_data.to(self.device))
        hparams_delta = self.model.lr(example_data.to(self.device))
        hparams_delta = torch.mean(hparams_delta)

        self.model.loss = F.nll_loss(out, example_targets.to(self.device))

        # multiply/divide learning rate
        # differentiable learning rate INSIDE the optimizer
        # just do SGD and manual parameter

        return out, torch.exp(hparams_delta)


    def selflearn(self, model=None, optimizer_class=None):


        if model is None:
            model = self.model
        if optimizer_class is None:
            optimizer_class = self.optimizer_cls

        out, hparams_delta = self.inference_step()


        # hparams_old = self.hparams_old
        # hparams_total = hparams_old + hparams_delta



        self.lr_out = hparams_delta#hparams_total

        # parameters [learning rate here is] can be learnable in pytorch
        hparams_dict = {'lr': 0}

        if 'optimizer' not in self.internals:
            self.internals['optimizer'] = optimizer_class(list(model.parameters()), **hparams_dict)
        # optimizer_hyperparams = ...
        # abstract, redundant element criteria: can compute if left out (prior allows to)
        # can remove those for caching

        optim = self.internals['optimizer']

        for g in optim.param_groups:
            g['lr'] = self.lr_out

        self.internals['optimizer'].zero_grad()

        model.loss.backward() # or loss(models | globals={'loss': model.loss or config[model.loss]})

        self.internals['optimizer'].step()
