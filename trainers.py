import copy
import time
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import tqdm

from utils import get_default_device, standard_normalization
from utils import RunningStats


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer, 
                 criterion, 
                 name: str,
                 bag_of_ops,
                 agent=None,
                 M=8,
                 rl_steps=16,
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

        self.agent = agent
        self.rl_steps = rl_steps
        self.deprecation_rate = deprecation_rate
        self.n_transforms = None if not agent else agent.n_transforms

        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.M = M
        self.random_erasing = torchvision.transforms.RandomErasing(
            p=1, scale=(0.25, 0.25), ratio=(1., 1.))

        self.normalize = normalize
        self.writer = SummaryWriter(f'runs/{name}')
        self.stats = RunningStats()

    def fit(self, train_dataloader, test_dataloader=None, 
            n_epochs=200, scheduler=None):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input # output
            return hook
        hook = self.model.fc.register_forward_hook(get_activation('fc'))

        for epoch in range(n_epochs):
            if self.agent:
                losses = torch.zeros(
                    (self.M,), dtype=torch.float, device=self.device)

                # update bins
                bins_diff = self.agent.max_bins - self.agent.min_bins
                bins = min(1, 2 * (epoch+1) / n_epochs)
                bins = int(self.agent.min_bins + bins*bins_diff)
                if self.agent.get_bins() != bins:
                    self.agent.reset_bins(bins)

                # random actions
                actions = self.agent.act(self.M)
                prob = max(0, 0.5 * (1 - 2 * epoch / n_epochs))

                if prob > 0:
                    is_rand = torch.rand(self.M, 1, 1, device=actions.device)
                    is_rand = (is_rand < prob).float()
                    rand_actions = torch.rand(*actions.size(),
                                              device=actions.device)
                    actions = (1-is_rand)*actions + is_rand*rand_actions

                policies = [self.agent.decode_policy(action, resize=False)
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

                if self.agent:
                    mini_size = batch_size // self.M

                    _xs = xs
                    xs = torch.cat(
                        [policy(xs[i*mini_size:(i+1)*mini_size])
                            for i, policy in enumerate(policies)],
                        dim=0)

                    prob = max(0, 0.5 * (1 - 2 * epoch / n_epochs))
                    if prob > 0:
                        is_easy = torch.rand(xs.size(0), 1, 1, 1, 
                                             device=xs.device)
                        is_easy = (is_easy < prob).float()
                        xs = is_easy*_xs + (1-is_easy)*xs # temp

                    xs = self.random_erasing(xs)

                if self.normalize:
                    xs = self.normalize(xs)
                    org_xs = self.normalize(org_xs)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.model(xs)
                    loss = self.criterion(ys_hat, ys)

                    loss[:batch_size].mean().backward()
                    self.optimizer.step()

                if self.agent:
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

                    loss = loss.detach()
                    losses = 0.95 * losses + 0.05 * loss.reshape(self.M, -1).mean(1)

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

            if self.agent:
                # summarize the epoch
                mean = self.stats.mean().square().sum(-1).sqrt()
                std = self.stats.std().square().sum(-1).sqrt()
                coef = mean / std.clamp(min=1e-8)

                self.writer.add_histogram('mean', mean, epoch)
                self.writer.add_histogram('std', std, epoch)
                self.writer.add_histogram('coef', coef, epoch)
                self.writer.add_histogram('losses', losses, epoch)

                coef = standard_normalization(coef)
                losses = standard_normalization(losses)

                rewards = - losses # - coef
                self.agent.cache_batch(actions, rewards)
                self.agent.learn(self.rl_steps)

                # save image
                action = self.agent.act(1, enable_dropout=False)[0]
                action = self.agent.preprocess_actions(action)
                n_ops = action.size(-2)
                self.writer.add_image('output', action.transpose(0, 1), 
                                      epoch, dataformats='HW')

                zeros = torch.mean(action[-2, -self.n_transforms:]) * n_ops
                self.writer.add_scalar('diag/zeros', zeros, epoch)
                print(f'zeros: {zeros}')

                bins = action.size(-1) - self.n_transforms
                danger_mag = torch.sum(action[-1, :int(0.75 * bins)])
                danger_prob = torch.mean(action[-1, -self.n_transforms:]) * n_ops
                self.writer.add_scalar('diag/danger_mag', danger_mag, epoch)
                self.writer.add_scalar('diag/danger_prob', danger_prob, epoch)
                print(f'danger: ({danger_mag}, {danger_prob})')

                acts = self.agent.act(128)
                ent = (acts - acts.mean(0, keepdim=True)).abs().mean(0)
                self.writer.add_scalar('diag/ent', ent.mean(), epoch)
                self.writer.add_image('imgs/ent', ent.transpose(0, 1), 
                                      epoch, dataformats='HW')
                print(f'ent: {ent.mean()}, {ent.min()}, {ent.max()}')
                
                # corr
                actions = actions - actions.mean(0, keepdim=True) # [batch, n_ops, k]
                weights = torch.matmul(
                    F.normalize(actions, dim=0).transpose(0, -1),
                    F.normalize(rewards, dim=0)) # [k, n_ops]
                self.writer.add_image('imgs/abs_corr', weights.abs(),
                                      epoch, dataformats='HW')

        self.writer.add_hparams(
            {'hp/M': self.M, 'hp/rl_steps': self.rl_steps},
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

