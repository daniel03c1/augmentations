import copy
import time
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter


class ClassificationTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer, 
                 criterion, 
                 name: str,
                 device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.name = name

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = self.model.to(self.device)

    def fit(self, train_dataloader, val_dataloader, n_epochs):
        for epoch in range(n_epochs):
            print(self.run(train_dataloader, train=True))
            print(self.run(val_dataloader, train=False))
            self.save()

    def run(self, dataloader, train=False, scheduler=None):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_acc = 0
        count = 0 # count the number of samples

        for xs, ys in tqdm.tqdm(dataloader):
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                ys_hat = self.model(xs)
                loss = self.criterion(ys_hat, ys)

                if train:
                    loss.backward()
                    self.optimizer.step()

            pred_cls = torch.argmax(ys_hat, -1)
            total_loss += loss.item() * xs.size(0)
            total_acc += torch.sum(pred_cls == ys.data)
            count += xs.size(0)

        if train and scheduler is not None:
            scheduler.step()

        return total_loss/count, total_acc/count

    def save(self):
        torch.save(self.model.state_dict(), self.name)

