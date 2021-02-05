import copy
import time
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter


class BasicTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer, 
                 criterion, 
                 name: str,
                 metrics=None, 
                 device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.name = name

        if metrics is None:
            metrics = []
        self.metrics = []
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = self.model.to(self.device)

    def fit(self, train_dataloader, val_dataloader, n_epochs):
        for epoch in range(n_epochs):
            print(self.train(train_dataloader))
            print(self.eval(val_dataloader))
            self.save()

    def train(self, dataloader):
        self.model.train()

        total_loss = 0
        count = 0
        for xs, ys in dataloader:
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            self.optimizer.zero_grad()

            ys_hat = self.model(xs)
            loss = self.criterion(ys_hat, ys)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / count

    def eval(self, dataloader):
        self.model.eval()

        total_loss = 0
        count = 0
        with torch.no_grad():
            for xs, ys in dataloader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)

                ys_hat = self.model(xs)
                loss = self.criterion(ys_hat, ys)

                total_loss += loss.item()
                count += 1

        return total_loss / count

    def save(self):
        torch.save(self.model.state_dict(), self.name)

