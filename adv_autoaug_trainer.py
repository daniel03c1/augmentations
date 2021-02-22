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
                 bag_of_ops,
                 rl_agent=None,
                 M=8,
                 rl_n_steps=16,
                 device=None):
        if device is None:
            device = get_default_device()
        self.device = device

        # main model
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.name = name
        self.best_acc = 0

        self.rl_agent = rl_agent
        self.rl_n_steps = rl_n_steps

        self.bag_of_ops = bag_of_ops
        self.M = M
        self.random_erasing = torchvision.transforms.RandomErasing(
            p=0.5, scale=(0.5, 0.5), ratio=(1., 1.))
        self.normalize = transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        self.op_visit_count = torch.zeros([self.bag_of_ops.n_ops]*2,
                                          dtype=torch.int)

    def fit(self, 
            train_dataloader, val_dataloader=None, 
            n_epochs=200, scheduler=None):
        chan_axis = 1
        '''
        self.activation = {}
        # for intermediate outputs
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = input # output
            return hook
        hook = self.model.fc.register_forward_hook(get_activation('avgpool'))
        '''

        for epoch in range(n_epochs):
            # for RL
            if self.rl_agent:
                losses = torch.zeros(
                    (self.M-1,), dtype=torch.float, device=self.device)
                states = torch.zeros(
                    (self.M-1, 1), dtype=torch.long, device=self.device)
                policies = self.rl_agent.act(states)
                prior_probs = policies
                policies = [self.decode_policy(policy) for policy in policies]
                print(self.op_visit_count)

            '''
            op_grads = torch.zeros(
                (self.M-1,), dtype=torch.float, device=self.device)
            '''
            self.model.train()

            total_loss = 0
            total_acc = 0
            count = 0 # count the number of samples
            start = time.time()

            for xs, ys in train_dataloader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                
                # multiply data
                multiply = 1
                xs = xs.repeat(multiply, 1, 1, 1)
                ys = ys.repeat(multiply)

                # subdivide into M parts
                batch_size = xs.size(0)
                if self.rl_agent:
                    mini_size = batch_size // self.M
                    xs = torch.cat(
                        [policy(xs[i*mini_size:(i+1)*mini_size]) 
                         for i, policy in enumerate(policies)] \
                        + [xs[-mini_size:]],
                        dim=0)

                xs = self.random_erasing(xs)
                xs = self.normalize(xs)

                # training
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys)
                    assert ys_hat.max() < 10000.

                    loss.mean().backward()
                    self.optimizer.step()

                '''
                # gradients & losses for RL
                inter = self.activation['avgpool'][0] # .squeeze()
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

                '''
                if self.rl_agent:
                    loss = loss.detach()
                    losses = 0.95 * losses \
                           + 0.05 * loss.reshape(self.M, -1).mean(1)[:-1]

                # metrics
                pred_cls = torch.argmax(ys_hat, -1)
                total_loss += loss.sum()
                total_acc += torch.sum(pred_cls == ys.data)
                count += xs.size(0)

            # verbose
            total_loss = total_loss / count
            total_acc = total_acc / count
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'{epoch}({time.time()-start:.3f})-lr:{current_lr:.6f} '
                  f'loss: {total_loss.item():.5f} acc: {total_acc.item():.5f}')
            if val_dataloader is not None:
                val_loss, val_acc = self.eval(val_dataloader)
                print(f'validation | loss: {val_loss.item():.5f}, '
                        f'acc: {val_acc.item():.5f}')
                if val_acc.item() > self.best_acc:
                    self.best_acc = val_acc.item()

            if scheduler:
                scheduler.step()

            if self.rl_agent:
                # calculate rewards
                rewards = losses

                # normalize rewards for stable training
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                rewards = rewards.view(-1, 1, 1) # TODO: remove this
                self.rl_agent.cache_batch(states, prior_probs, rewards, states)
                self.rl_agent.learn(n_steps=self.rl_n_steps)
                self.rl_agent.save()

                print('losses: ', losses)
                # print(op_grads)
                print('rewards: ', rewards.squeeze())

        # hook.remove() 
        self.save()
        print(f"Best ACC: {self.best_acc:.5f}")

    def eval(self, dataloader):
        self.model.eval()

        total_loss = 0
        total_acc = 0
        count = 0 # count the number of samples

        with torch.no_grad():
            for xs, ys in dataloader: 
                xs = xs.to(self.device)
                ys = ys.to(self.device)

                xs = self.normalize(xs)

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


def aug_policies(states, prior_probs, rewards, next_states):
    prior_probs = prior_probs * 1 # copy original prior_probs
    n_subpolicies = prior_probs.size(-2) // 4

    for i in range(states.size(0)):
        new_probs = prior_probs[i].view(n_subpolicies, 4, -1)
        new_probs = new_probs[torch.randperm(5)]
        prior_probs[i] = new_probs.view(n_subpolicies*4, -1)

    return states, prior_probs, rewards, next_states


if __name__ == '__main__':
    import torch.nn as nn
    from torch import optim

    from agents import PPOAgent
    from dataloader import EfficientCIFAR10
    from models import Controller
    from transforms import transforms as bag_of_ops
    from wideresnet import WideResNet

    # transforms
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomCrop(32, padding=4), # , padding_mode='reflect'),
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
    model = WideResNet(28, 10, 0.3, n_classes=10)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True,
                          weight_decay=5e-4)

    # RL
    M = 4
    c = Controller(output_size=bag_of_ops.n_ops)
    c_optimizer = optim.Adam(c.parameters(), lr=0.00035)
    ppo = PPOAgent(c, name='ppo.pt', batch_size=M-1, augmentation=aug_policies)

    trainer = AdvAutoaugTrainer(model=model,
                                optimizer=optimizer,
                                criterion=criterion,
                                name='test.pt',
                                bag_of_ops=bag_of_ops,
                                rl_n_steps=8, # 16,
                                M=M, 
                                rl_agent=ppo)

    print(bag_of_ops.ops)
    n_epochs = 200
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[60,120,160],
                                                     gamma=0.2)
    trainer.fit(dataloaders['train'], dataloaders['val'], 
                n_epochs=n_epochs,
                scheduler=scheduler)

