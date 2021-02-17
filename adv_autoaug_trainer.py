import copy
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import get_default_device


class AdvAutoaugTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer, 
                 criterion, 
                 name: str,
                 rl_agent,
                 bag_of_ops,
                 device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.name = name

        self.rl_agent = rl_agent
        self.bag_of_ops = bag_of_ops

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = self.model.to(self.device)
        self.best_acc = 0

        self.op_visit_count = torch.zeros([self.bag_of_ops.n_ops]*2,
                                          dtype=torch.int)

    def fit(self, 
            train_dataloader, val_dataloader=None, 
            n_epochs=200, scheduler=None):
        chan_axis = 1
        M = 8
        
        # for intermediate outputs
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input # output
            return hook
        avg = self.model.fc.register_forward_hook(get_activation('avgpool'))

        for epoch in range(n_epochs):
            losses = torch.zeros(
                (M-1,), dtype=torch.float, device=self.device)
            op_grads = torch.zeros(
                (M-1,), dtype=torch.float, device=self.device)
            states = torch.zeros((M-1, 1), dtype=torch.long, device=self.device)
            policies = self.rl_agent.act(states)
            prior_probs = policies
            policies = [self.decode_policy(policy) for policy in policies]
            print(self.op_visit_count)

            total_loss = 0
            total_acc = 0
            count = 0 # count the number of samples
            start = time.time()

            for xs, ys in train_dataloader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                batch_size = xs.size(0)
                
                xs = torch.cat([policy(xs) for policy in policies]+[xs], dim=0)
                ys = ys.repeat(M)

                self.optimizer.zero_grad()

                # training
                with torch.set_grad_enabled(True):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys)
                    assert ys_hat.max() < 1000.

                    loss.mean().backward()
                    self.optimizer.step()

                # gradients & losses for RL
                inter = activation['avgpool'][0] # .squeeze()
                grads = torch.autograd.grad(
                    self.criterion(self.model.fc(inter), ys).mean(),
                    inter)[0].detach()
                base = grads[-batch_size:]
                base_mag = torch.dist(base, base * 0)

                for i in range(M-1):
                    op_mag = torch.dist(grads[batch_size*i:batch_size*(i+1)], 
                                        base) 
                    op_mag /= base_mag # normalize (magnitude keep rises)
                    op_grads[i] = 0.95*op_grads[i] + 0.05*op_mag

                loss = loss.detach()
                losses = 0.95*losses + 0.05*loss.reshape(M, -1).mean(1)[:-1]

                # metrics
                pred_cls = torch.argmax(ys_hat, -1)
                total_loss += loss.sum()
                total_acc += torch.sum(pred_cls == ys.data)
                count += xs.size(0)

            # verbose
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

            # calculate rewards
            # rewards = (losses - losses.mean()) / (losses.std() + 1e-8) 
            # rewards = calculate_rewards(losses, op_grads)
            base_loss = loss[-batch_size:].mean()
            rewards = - 2*op_grads + losses / base_loss

            # normalize rewards for stable training
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = rewards.view(-1, 1, 1) # TODO: remove this
            self.rl_agent.cache_batch(states, prior_probs, rewards, states)
            self.rl_agent.learn(n_steps=16)
            self.rl_agent.save()

            print(losses / base_loss - 1)
            print(op_grads)
            print(rewards)

        avg.remove() # detach hook

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

    def decode_policy(self, policy, visit_count=True):
        n_subpolicies = policy.size(-2) // 4
        n_magnitudes = policy.size(-1)

        probs = policy
        policy = torch.multinomial(policy, 1).squeeze(-1)
        policy = [policy[i*4:(i+1)*4]
                  for i in range(n_subpolicies)]
        if visit_count:
            for i in range(n_subpolicies):
                op1, mag1, op2, mag2 = policy[i]

                self.op_visit_count[op1][mag1] += 1
                self.op_visit_count[op2][mag2] += 1

        policy = [
            transforms.Compose([
                self.bag_of_ops[op1]((mag1+1)/n_magnitudes),
                self.bag_of_ops[op2]((mag2+1)/n_magnitudes)
            ]) 
            for op1, mag1, op2, mag2 in policy]

        return transforms.RandomChoice(policy)


def calculate_rewards(losses, grad_diffs):
    # 1 for pareto-efficient data points else 0
    # want to maximize losses while maintaining small grad diff
    rewards = losses * 0
    indices = torch.argsort(grad_diffs)        
    max_losses = losses[indices[0]]
    for i in indices:
        loss = losses[i]
        if loss >= max_losses:
            rewards[i] = 1
            max_losses = loss
    return rewards


if __name__ == '__main__':
    import torch.nn as nn
    from torch import optim
    from dataloader import EfficientCIFAR10
    from models import Controller
    from transforms import transforms as bag_of_ops
    from wide_resnet import WideResNet
    from agents import PPOAgent

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
    PATH = '/datasets/datasets/cifar'
    for mode in ['train', 'val']:
        dataloaders[mode] = EfficientCIFAR10(PATH,
                                             train=mode == 'train',
                                             transform=data_transforms[mode])
        dataloaders[mode] = torch.utils.data.DataLoader(
            dataloaders[mode],
            batch_size=128,
            shuffle=mode=='train',
            num_workers=12)

    # model
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    # model = WideResNet(28, 10, 0.3, num_classes=10)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4)

    # RL
    c = Controller(output_size=bag_of_ops.n_ops)
    c_optimizer = optim.Adam(c.parameters(), lr=0.00035)
    def c_aug(states, prior_probs, rewards, next_states):
        prior_probs = prior_probs * 1 # copy original prior_probs
        n_subpolicies = prior_probs.size(-2) // 4

        for i in range(states.size(0)):
            new_probs = prior_probs[i].view(n_subpolicies, 4, -1)
            new_probs = new_probs[torch.randperm(5)]
            prior_probs[i] = new_probs.view(n_subpolicies*4, -1)

        return states, prior_probs, rewards, next_states
    ppo = PPOAgent(c, batch_size=7, augmentation=c_aug)

    trainer = AdvAutoaugTrainer(model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                name='test.pt',
                                rl_agent=ppo,
                                bag_of_ops=bag_of_ops)

    print(bag_of_ops.ops)
    n_epochs = 200
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[120,160],
                                                     gamma=0.1)
    trainer.fit(dataloaders['train'], dataloaders['val'], 
                n_epochs=n_epochs,
                scheduler=scheduler)

