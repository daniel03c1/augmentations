import copy
import time
import torch
import torchvision
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

        self.op_visit_count = torch.zeros([self.bag_of_ops.n_ops]*2,
                                          dtype=torch.int)

    def fit(self, 
            train_dataloader, val_dataloader=None, 
            n_epochs=200, scheduler=None):
        chan_axis = 1
        M = 8
        for epoch in range(n_epochs):
            losses = torch.zeros((M,), dtype=torch.float, device=self.device)
            policies = self.generate_policies(M-1)
            prior_probs = policies
            policies = [self.decode_policy(policy) for policy in policies]

            total_loss = 0
            total_acc = 0
            count = 0 # count the number of samples
            start = time.time()

            for xs, ys in train_dataloader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                
                xs = torch.cat([policy(xs) for policy in policies]+[xs], dim=0)
                ys = ys.repeat(M)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys)
                    assert torch.isnan(loss).sum() == 0

                    loss.mean().backward()
                    self.optimizer.step()

                losses = 0.95*losses + 0.05*loss.detach().reshape(M, -1).mean(1)

                pred_cls = torch.argmax(ys_hat, -1)
                total_loss += loss.sum()
                total_acc += torch.sum(pred_cls == ys.data)
                count += xs.size(0)

            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'{epoch}({time.time()-start:.3f})-{current_lr:.6f} '
                  f'loss: {(total_loss/count).item()} '
                  f'acc: {(total_acc/count).item()}')
            if val_dataloader is not None:
                val_loss, val_acc = self.eval(val_dataloader)
                print(f'validation | loss: {val_loss.item()}, '
                      f'acc: {val_acc.item()}')

            if scheduler:
                scheduler.step()

            # losses = (losses - losses.mean()) / (losses.std() + 1e-8) # across M
            # self.controller_update(prior_probs, losses, n_steps=20)
            print(self.op_visit_count)

    def eval(self, dataloader):
        self.model.eval()

        total_loss = 0
        total_acc = 0
        count = 0 # count the number of samples

        for xs, ys in dataloader: 
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
        x = torch.zeros((m, 1), dtype=torch.long, device=self.device)
        return self.controller(x)

    def decode_policy(self, policy, visit_count=True):
        n_subpolicies = policy.size(-2) // 4
        n_magnitudes = policy.size(-1)

        # policy = torch.argmax(policy, -1) # [subpolicy*4]
        policy = torch.randint(high=n_magnitudes,
                               size=policy.size()[:-1],
                               dtype=torch.long,
                               device=policy.device)

        policy = [policy[i*4:(i+1)*4]
                  for i in range(n_subpolicies)]
        if visit_count:
            for op1, mag1, op2, mag2 in policy:
                self.op_visit_count[op1][mag1] += 1
                self.op_visit_count[op2][mag2] += 1

        policy = [
            transforms.Compose([
                self.bag_of_ops[op1]((mag1+1)/n_magnitudes),
                self.bag_of_ops[op2]((mag2+1)/n_magnitudes)
            ]) 
            for op1, mag1, op2, mag2 in policy]

        return transforms.RandomChoice(policy)

    def controller_update(self, prior_probs, rewards, n_steps=10):
        torch.autograd.set_detect_anomaly(True)
        states = torch.zeros((rewards.size(0), 1), 
                             dtype=torch.long,
                             device=self.device)
        prior_log_probs = prior_probs.detach().log()
        eps = 0.2

        for i in range(n_steps):
            self.controller_opt.zero_grad()

            probs = self.controller(states)
            ratios = torch.exp(probs.log() - prior_log_probs)

            loss = -torch.min(ratios*rewards,
                              torch.clamp(ratios, 1-eps, 1+eps)*rewards)
            loss += 1e-5 * (probs * probs.log()).sum(-1).mean()

            torch.mean(loss).backward()
            self.controller_opt.step()


if __name__ == '__main__':
    import torch.nn as nn
    from torch import optim
    from models import Controller
    from transforms import transforms as bag_of_ops
    from dataloader import EfficientCIFAR10
    from wide_resnet import WideResNet

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
        dataloaders[mode] = EfficientCIFAR10('/datasets/datasets/cifar', 
                                             train=mode == 'train',
                                             transform=data_transforms[mode])
        dataloaders[mode] = torch.utils.data.DataLoader(
            dataloaders[mode],
            batch_size=128,
            shuffle=mode=='train',
            num_workers=12)

    # model
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    # model = WideResNet(28, 10, 0.3, num_classes=10)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4)

    c = Controller(output_size=bag_of_ops.n_ops)
    c_optimizer = optim.Adam(c.parameters(), lr=0.00035)

    trainer = AdvAutoaugTrainer(model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                name='test.pt',
                                controller=c,
                                controller_opt=c_optimizer,
                                controller_name='c_test.pt',
                                bag_of_ops=bag_of_ops)

    n_epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           n_epochs)
    trainer.fit(dataloaders['train'], dataloaders['val'], 
                n_epochs=n_epochs,
                scheduler=scheduler)

