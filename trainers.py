import copy
import time
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import get_default_device


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer, 
                 criterion, 
                 name: str,
                 bag_of_ops,
                 rl_agent=None,
                 M=8,
                 rl_n_steps=16,
                 normalize=None,
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
            p=1, scale=(0.25, 0.25), ratio=(1., 1.))

        self.normalize = normalize
        self.writer = SummaryWriter(f'runs/{name}')

    def fit(self, train_dataloader, test_dataloader=None, 
            n_epochs=200, scheduler=None):
        chan_axis = 1

        for epoch in range(n_epochs):
            # for RL
            if self.rl_agent:
                losses = torch.zeros(
                    (self.M,), dtype=torch.float, device=self.device)
                states = torch.zeros(self.M)
                rand_prob = max(0, 0.5 - epoch / (n_epochs//4))
                actions, dist = self.rl_agent.act(states, rand_prob)
                policies = [self.rl_agent.decode_policy(action) 
                            for action in actions]

            self.model.train()

            total_loss = 0
            total_acc = 0
            count = 0 # count the number of samples
            start = time.time()

            for xs, ys in train_dataloader:
                xs = xs.to(self.device)
                ys = ys.to(self.device)
                
                # subdivide into M parts
                batch_size = xs.size(0)
                if self.rl_agent:
                    mini_size = batch_size // self.M

                    org_step = 8
                    org_xs = xs[::org_step]

                    xs = torch.cat(
                        [policy(xs[i*mini_size:(i+1)*mini_size]) 
                            for i, policy in enumerate(policies)],
                        dim=0)
                    xs = self.random_erasing(xs)

                if self.normalize:
                    xs = self.normalize(xs)

                # training
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys)

                    loss[:batch_size].mean().backward()
                    self.optimizer.step()

                if self.rl_agent:
                    # original losses
                    loss = loss.detach()
                    losses = 0.95 * losses \
                           + 0.05 * loss.reshape(self.M, -1).mean(1)

                # metrics
                pred_cls = torch.argmax(ys_hat, -1)
                total_loss += loss.sum()
                total_acc += torch.sum(pred_cls == ys.data)
                count += xs.size(0)

            # verbose
            total_loss = total_loss / count
            total_acc = total_acc / count
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'{epoch:3}({time.time()-start:.3f})-lr:{current_lr:.6f} '
                  f'loss: {total_loss.item():.5f}, acc: {total_acc.item():.5f}')
            self.writer.add_scalar('train/loss', total_loss, epoch)
            self.writer.add_scalar('train/acc', total_acc, epoch)
            self.writer.add_scalar('hp/lr', current_lr, epoch)

            if test_dataloader is not None:
                test_loss, test_acc = self.eval(test_dataloader)
                print(f'{" "*11}validation | loss: {test_loss.item():.5f}, '
                        f'acc: {test_acc.item():.5f}')
                self.writer.add_scalar('test/loss', test_loss, epoch)
                self.writer.add_scalar('test/acc', test_acc, epoch)
                if test_acc.item() > self.best_acc:
                    self.best_acc = test_acc.item()
                    self.save()

            if scheduler:
                scheduler.step()

            if self.rl_agent:
                self.writer.add_histogram('losses', losses, epoch)
                rewards = -losses
                assert torch.isnan(rewards).sum() == 0

                # normalize rewards for stable training
                rewards = rewards.view(-1, 1, 1) # TODO: remove this
                self.rl_agent.cache_batch(
                    states, actions, dist, rewards, states)
                self.rl_agent.learn(n_steps=self.rl_n_steps)
                self.rl_agent.save()

                self.writer.add_histogram('rewards', rewards, epoch)
                _, output = self.rl_agent.act(torch.zeros(1))
                self.writer.add_image('output', output, epoch)

        self.writer.add_hparams(
            {'hp/M': self.M, 'hp/rl_n_steps': self.rl_n_steps},
            {'hp/best_acc': self.best_acc})

        hook.remove() 
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

                if self.normalize:
                    xs = self.normalize(xs)

                with torch.set_grad_enabled(False):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys).mean()

                pred_cls = torch.argmax(ys_hat, -1)
                total_loss += loss * xs.size(0)
                total_acc += torch.sum(pred_cls == ys.data)
                count += xs.size(0)

        return total_loss/count, total_acc/count

    def save(self, name=None):
        if not name:
            name = self.name
        if not name.endswith('.pt'):
            name += '.pt'
        torch.save(self.model.state_dict(), name)

