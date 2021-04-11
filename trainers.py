import copy
import time
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import get_default_device
from utils import RunningStats


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
                 deprecation_rate=1.,
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
        self.deprecation_rate = deprecation_rate

        self.bag_of_ops = bag_of_ops
        self.M = M
        self.random_erasing = torchvision.transforms.RandomErasing(
            p=1, scale=(0.25, 0.25), ratio=(1., 1.))

        self.normalize = normalize
        self.writer = SummaryWriter(f'runs/{name}')
        self.stats = RunningStats()

    def fit(self, train_dataloader, test_dataloader=None, 
            n_epochs=200, scheduler=None):
        chan_axis = 1

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input # output
            return hook
        hook = self.model.fc.register_forward_hook(get_activation('fc'))

        self.stats.clear()

        for epoch in range(n_epochs):
            # for RL
            if self.rl_agent:
                losses = torch.zeros(
                    (self.M,), dtype=torch.float, device=self.device)
                states = torch.zeros(self.M)
                rand_prob = max(0, 1 - epoch / (n_epochs//4))
                actions = self.rl_agent.act(states, rand_prob)

                # for test
                # from torchsummary import summary
                # summary(self.rl_agent.net, states.shape)

                policies = [self.rl_agent.decode_policy(action) 
                            for action in actions]

                self.stats.clear()

            total_loss = 0
            total_acc = 0
            count = 0 # count the number of samples
            start = time.time()

            for xs, ys in train_dataloader:
                self.model.train()

                xs = xs.to(self.device)
                ys = ys.to(self.device)
                
                # subdivide into M parts
                batch_size = xs.size(0)

                org_step = 8
                org_xs = xs[::org_step]

                if self.rl_agent:
                    mini_size = batch_size // self.M

                    xs = torch.cat(
                        [policy(xs[i*mini_size:(i+1)*mini_size]) 
                            for i, policy in enumerate(policies)],
                        dim=0)
                    xs = self.random_erasing(xs)

                if self.normalize:
                    xs = self.normalize(xs)
                    org_xs = self.normalize(org_xs)

                # training
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys)

                    loss[:batch_size].mean().backward()
                    self.optimizer.step()

                if self.rl_agent:
                    augs = activation['fc'][0].detach()[:batch_size:org_step]

                    self.model.eval()
                    with torch.set_grad_enabled(False):
                        _ = self.model(org_xs)
                    org = activation['fc'][0].detach()

                    # augs: [batch_size//org_step, h_dim]
                    # -> [M, batch_size//org_step//M, h_dim]
                    dists = (augs - org).reshape(self.M, -1, augs.size(-1))
                    for i in range(dists.size(1)):
                        self.stats.push(dists[:, i])

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

            if test_dataloader:
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
                mean = self.stats.mean().square().sum(-1).sqrt()
                std = self.stats.std().square().sum(-1).sqrt()
                coef = mean / (std + 1e-8)

                self.writer.add_histogram('mean', mean, epoch)
                self.writer.add_histogram('std', std, epoch)
                self.writer.add_histogram('coef', coef, epoch)
                self.writer.add_histogram('losses', losses, epoch)

                losses = (losses - losses.mean()) / (losses.std() + 1e-8)
                # coef = (coef - coef.mean()) / (coef.std() + 1e-8)

                rewards = -losses # - 4*coef
                assert torch.isnan(rewards).sum() == 0

                # normalize rewards for stable training
                rewards = rewards.view(-1, 1, 1, 1) # TODO: remove this
                self.rl_agent.cache_batch(
                    states, actions, rewards, states)
                self.rl_agent.learn(n_steps=self.rl_n_steps)
                self.rl_agent.apply_gamma(self.deprecation_rate)
                self.rl_agent.save()

                # [layer, ops, bins+1]
                actions = self.rl_agent.act(torch.zeros(1), train=False)[0] 
                image = actions.transpose(-1, -2) # [layer, bins+1, ops]
                image = image.reshape(-1, image.size(-1))
                self.writer.add_image('output', image, epoch, dataformats='HW')

                n_ops = actions.size(-2)
                zeros = torch.mean(actions[:, -2, -1]) * n_ops
                bins = actions.size(-1) - 1
                danger_mag = torch.sum(actions[:, -1, :int(0.75 * bins)], -1)
                danger_mag = danger_mag.mean()
                danger_prob = torch.mean(actions[:, -1, -1]) * n_ops
                self.writer.add_scalar('diag/zeros', zeros, epoch)
                self.writer.add_scalar('diag/danger_mag', danger_mag, epoch)
                self.writer.add_scalar('diag/danger_prob', danger_prob, epoch)
                print(f'zeros: {zeros}, danger: ({danger_mag}, {danger_prob})')

        self.writer.add_hparams(
            {'hp/M': self.M, 'hp/rl_n_steps': self.rl_n_steps},
            {'hp/best_acc': self.best_acc})

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

