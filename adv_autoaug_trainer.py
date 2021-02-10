import copy
import time
import torch
import torchvision
import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


class AdvAutoaugTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer, 
                 criterion, 
                 name: str,
                 controller: torch.nn.Module,
                 controller_opt,
                 controller_name: str,
                 bag_of_ops,
                 device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.name = name

        self.controller = controller
        self.controller_opt = controller_opt
        self.controller_name = controller_name
        self.bag_of_ops = bag_of_ops

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = self.model.to(self.device)
        self.controller = self.controller.to(self.device)
        self.best_acc = 0

    def fit(self, train_dataloader, val_dataloader=None, n_epochs=200):
        chan_axis = 1
        M = 8
        for epoch in range(n_epochs):
            losses = torch.zeros((M,)).to(self.device)
            policies = self.generate_policies(M)

            total_loss = 0
            total_acc = 0
            count = 0 # count the number of samples
            start = time.time()

            for xs, ys in train_dataloader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                
                # applys M policies
                xs = torch.cat([policy(xs) for policy in policies], dim=0)
                ys = ys.repeat(M)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys)

                    loss.mean().backward()
                    self.optimizer.step()

                losses *= 0.9*losses + 0.1*loss.reshape(M, -1).mean(1)

                pred_cls = torch.argmax(ys_hat, -1)
                total_loss += loss.sum()
                total_acc += torch.sum(pred_cls == ys.data)
                count += xs.size(0)

            print(f'{time.time()-start:.3f}', total_loss/count, total_acc/count)
            if val_dataloader is not None:
                print(self.eval(val_dataloader))

            # losses = norm(losses) # norm across M
            # controller.update(losses) # REINFORCE

    def eval(self, dataloader):
        self.model.eval()

        total_loss = 0
        total_acc = 0
        count = 0 # count the number of samples

        for xs, ys in tqdm.tqdm(dataloader):
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            with torch.set_grad_enabled(False):
                ys_hat = self.model(xs)
                loss = self.criterion(ys_hat, ys).mean()

            pred_cls = torch.argmax(ys_hat, -1)
            total_loss += loss * xs.size(0)
            total_acc += torch.sum(pred_cls == ys.data)
            count += xs.size(0)

        return total_loss/count, total_acc/count

    def save(self):
        torch.save(self.model.state_dict(), self.name)

    def generate_policies(self, m):
        x = torch.zeros((m, 1), dtype=torch.long).to(self.device)
        policies = self.controller(x)
        return [self.decode_policy(policies[i]) for i in range(m)]

    def decode_policy(self, policy):
        n_subpolicies = policy.size(-2) // 4
        n_magnitudes = policy.size(-1)

        policy = torch.argmax(policy, -1) # [subpolicy*4]
        policy = [policy[i*4:(i+1)*4]
                  for i in range(n_subpolicies)]
        policy = [
            transforms.Compose([
                self.bag_of_ops[op1]((mag1+1)/n_magnitudes),
                self.bag_of_ops[op2]((mag2+1)/n_magnitudes)
            ]) 
            for op1, mag1, op2, mag2 in policy]

        return transforms.RandomChoice(policy)


if __name__ == '__main__':
    import torch.nn as nn
    from torch import optim
    from models import Controller
    from transforms import transforms as bag_of_ops
    from dataloader import EfficientCIFAR10

    # transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ]),
        'val': transforms.Compose([
        ]),
    }

    # datasets & dataloaders
    dataloaders = {}
    for mode in ['train', 'val']:
        dataloaders[mode] = EfficientCIFAR10('/media/data1/datasets/cifar', 
                                             train=mode == 'train',
                                             transform=data_transforms[mode])
        dataloaders[mode] = torch.utils.data.DataLoader(
            dataloaders[mode],
            batch_size=32,
            shuffle=mode=='train',
            num_workers=4)

    # model
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    c = Controller(output_size=bag_of_ops.n_ops)
    c_optimizer = optim.SGD(c.parameters(), lr=0.001, momentum=0.9)

    trainer = AdvAutoaugTrainer(model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                name='test.pt',
                                controller=c,
                                controller_opt=c_optimizer,
                                controller_name='c_test.pt',
                                bag_of_ops=bag_of_ops)
    trainer.fit(dataloaders['train'], dataloaders['val'], n_epochs=10)

