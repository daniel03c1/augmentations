import random
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *

DEVICE = get_default_device()


class DiscretePPOAgentv2:
    def __init__(self,
                 controller: nn.Module,
                 valuator: nn.Module,
                 name: str,
                 mem_maxlen: int,
                 batch_mem_maxlen: int,
                 lr=0.00035,
                 grad_norm=0.5,
                 batch_size=1,
                 epsilon=0.2,
                 ent_coef=1e-5,
                 **args):
        print(**args)
        self.device = get_default_device()
        self.controller = controller.to(self.device)
        self.valuator = valuator.to(self.device)

        self.name = name
        self.c_optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.v_optimizer = optim.Adam(self.valuator.parameters(), weight_decay=1e-5)
        self.v_criterion = nn.L1Loss()
        self.grad_norm = grad_norm

        self.sample_memory = deque(maxlen=mem_maxlen * batch_size)
        self.batch_memory = deque(maxlen=batch_mem_maxlen)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.ent_coef = ent_coef

    def act(self, states, train=True, **kwargs):
        if train:
            self.controller.train()
        else:
            self.controller.eval()
        return self.controller(states, **kwargs)

    def cache(self, action, reward):
        self.sample_memory.append((action.to(self.device).detach(),
                                   reward.to(self.device).detach()))

    def cache_batch(self, actions, rewards):
        for i in range(actions.size(0)):
            self.cache(actions[i], rewards[i])

        self.batch_memory.append((actions.to(self.device).detach(),
                                  rewards.to(self.device).detach()))

    def learn(self, n_steps=1):
        self.train_valuator()

        self.controller.train()
        for i in range(n_steps):
            self.c_optimizer.zero_grad()

            old_actions, rewards = self.recall()
            rewards = self.valuator(old_actions).squeeze(-1)
            rewards = standard_normalization(rewards)

            actions = self.controller([0]*rewards.size(0))

            # PPO
            ratios = self.controller.calculate_log_probs(actions) \
                   - self.controller.calculate_log_probs(old_actions).detach()
            rewards = rewards.view(-1, *([1]*(len(ratios.shape)-1)))

            loss = -torch.min(
                ratios*rewards,
                ratios.clamp(1-self.epsilon, 1+self.epsilon)*rewards)
            loss -= self.ent_coef * self.controller.calculate_entropy(actions)

            torch.mean(loss).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                           self.grad_norm)
            self.c_optimizer.step()

        return loss # the last loss

    def recall(self):
        batch = random.sample(self.sample_memory, self.batch_size)
        actions, rewards = map(torch.stack, zip(*batch))
        return actions, rewards

    def decode_policy(self, *args, **kwargs):
        return self.controller.decode_policy(*args, **kwargs)

    def train_valuator(self, tolerance=0.1):
        self.valuator.train()
        epochs = len(self.batch_memory)

        while True:
            total_loss = 0

            for _ in range(epochs):
                xs, ys = random.sample(self.batch_memory, 1)[0]
                actions = self.preprocess_actions(actions, resize=True)

                self.v_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.valuator(actions).squeeze(-1)
                    loss = self.v_criterion(standard_normalization(ys_hat), rewards)
                    loss.backward(retain_graph=True)
                    self.v_optimizer.step()

                    total_loss += (ys_hat.argsort() != rewards.argsort()).float().mean() / epochs
            if total_loss < tolerance:
                break
        self.valuator.eval()

    def preprocess_actions(self, 
                           action, 
                           normalize=True, 
                           resize=False, 
                           des_dim=None):
        magnitudes = action[..., :-self.n_transforms]
        probs = action[..., -self.n_transforms:]

        if des_dim is None:
            des_dim = self.max_bins
        if resize:
            magnitudes = linear_resize(magnitudes, des_dim)
        if normalize:
            magnitudes /= magnitudes.sum(-1, keepdim=True)
            probs /= probs.sum(-2, keepdim=True)

        return torch.cat([magnitudes, probs], -1)

    def decode_policy(self, action, *args, **kwargs):
        action = self.preprocess_actions(action.detach(), *args, **kwargs)
        return DiscreteRandomApply(self.bag_of_ops, action, self.n_transforms)


class BasicController(nn.Module):
    """ simple and general controller (discrete version) """
    def __init__(self, 
                 bag_of_ops, 
                 n_transforms=4, 
                 n_layers=3,
                 h_dim=128,
                 dropout_p=0.5,
                 bins=3,
                 **kwargs):
        super(BasicController, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.n_transforms = n_transforms
        self.h_dim = h_dim
        self.bins = bins

        ''' modules '''
        self.outdim = self.bins + self.n_transforms
        self.fixed_inputs = torch.randn((1, h_dim), requires_grad=True) \
                          / math.sqrt(h_dim)

        # middle
        self.n_layers = n_layers
        self.fcs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for i in range(n_layers)])
        self.layernorms = nn.ModuleList(
            [nn.BatchNorm1d(h_dim) for i in range(n_layers)])
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_p) for i in range(n_layers)])

        self.to_magnitude = nn.Linear(h_dim, self.n_ops * bins)
        self.to_prob = nn.Linear(h_dim, self.n_ops * n_transforms)

    def forward(self, x: torch.Tensor):
        n_samples = len(x)
        x = self.fixed_inputs.repeat(n_samples, 1)

        for i in range(self.n_layers):
            x = self.fcs[i](x)
            x = self.layernorms[i](x)
            x = x * x.sigmoid()
            x = self.dropouts[i](x)

        magnitudes = self.to_magnitude(x)
        magnitudes = magnitudes.reshape(-1, self.n_ops, self.bins)

        probs = self.to_prob(x)
        probs = probs.reshape(-1, self.n_ops, self.n_transforms)
        return torch.cat([magnitudes, probs], -1).sigmoid()

    def calculate_log_probs(self, actions):
        return actions.clamp(min=EPS).log()

    def calculate_entropy(self, actions):
        return (actions - actions.mean(0, keepdim=True)).abs()

    def reset_bins(self, bins):
        with torch.no_grad():
            weight = self.to_magnitude.weight # (n_ops * bins, h_dim)
            bias = self.to_magnitude.bias # (n_ops * bins)

            # weight update
            weight = weight.reshape(self.n_ops, self.bins, -1)
            weight = weight.transpose(-2, -1)
            weight = linear_resize(weight, bins)
            weight = weight.transpose(-2, -1)
            weight = weight.reshape(-1, weight.size(-1))
            self.to_magnitude.weight = nn.parameter.Parameter(
                weight, requires_grad=True)

            # bias update
            bias = bias.reshape(self.n_ops, self.bins)
            bias = linear_resize(bias, bins)
            bias = bias.reshape(-1)
            self.to_magnitude.bias = nn.parameter.Parameter(
                bias, requires_grad=True)

            self.bins = bins


class EvolveAgent:
    def __init__(self,
                 valuator: nn.Module,
                 bag_of_ops,
                 min_bins=3,
                 max_bins=11,
                 n_transforms=2,
                 max_population=32,
                 **kwargs):
        self.device = get_default_device()
        print(self.device)
        self.valuator = valuator.to(self.device)
        self.v_optimizer = optim.Adam(self.valuator.parameters(), weight_decay=1e-5)
        self.v_criterion = nn.L1Loss()

        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops

        self.min_bins = min_bins
        self.max_bins = max_bins
        self.n_transforms = n_transforms

        self.max_population = max_population
        self.population = [torch.rand(self.n_ops, min_bins+n_transforms).to(self.device)
                           for i in range(max_population)]
        self.subjects = self.population
        self.history = []

    def act(self, states, **kwargs):
        return random.sample(self.subjects, k=len(states))

    def update(self, actions, rewards, tolerance=0.1, max_children=None):
        self.history.append((actions, standard_normalization(rewards)))
        self.train_valuator(tolerance)

        self.population.extend(actions)
        self.population, scores = self.select_best(self.population,
                                                   top_k=self.max_population)

        if max_children is None:
            max_children = self.max_population * 4
        parents = (random.choices(self.population,
                                  weights=scores-scores.min(),
                                  k=max_children)
                   for _ in range(2))

        children = self.breed(*parents)
        children = self.mutate(children)
        self.subjects, _ = self.select_best(children, top_k=self.max_population)

    def train_valuator(self, tolerance=0.1):
        self.valuator.train()
        epochs = len(self.history)

        while True:
            total_loss = 0

            for _ in range(epochs):
                actions, rewards = random.sample(self.history, 1)[0]
                actions = self.preprocess_actions(actions, resize=True).to(self.device)
                rewards = rewards.to(self.device)

                self.v_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    ys_hat = self.valuator(actions)[..., 0]
                    loss = self.v_criterion(standard_normalization(ys_hat), rewards)
                    loss.backward(retain_graph=True)
                    self.v_optimizer.step()

                    total_loss += (ys_hat.argsort() != rewards.argsort()).float().mean() / epochs
            if total_loss < tolerance:
                break
        self.valuator.eval()

    def select_best(self, actions, top_k=1):
        with torch.set_grad_enabled(False):
            scores = self.valuator(self.preprocess_actions(actions, resize=True))
            scores = scores.squeeze(-1)

        good_pop = scores.argsort()[-top_k:]
        new_actions = [action for i, action in enumerate(actions)
                       if i in good_pop]
        scores = scores[good_pop.sort()[0]]
        return new_actions, scores

    def breed(self, left_parents, right_parents):
        pairs = list(zip(left_parents, right_parents))
        children = []
        for pair in pairs:
            left, right = pair
            des_dim = max(left.size(-1), right.size(-1)) - self.n_transforms
            left = self.preprocess_action(left, normalize=False, resize=True, des_dim=des_dim)
            right = self.preprocess_action(right, normalize=False, resize=True, des_dim=des_dim)

            coef = torch.rand(*left.size(), device=self.device)
            if random.random() < 0.5: # lin interpolation or binary
                coef = (coef < 0.5).float()
            child = coef*left + (1-coef)*right
            children.append(child)
        return children

    def mutate(self, actions, prob=0.1):
        new_actions = []
        for action in actions:
            if random.random() < prob:
                action = action / action.max() * (0.5 + 0.5*torch.rand(1, device=self.device)[0])

            coef = (torch.rand(*action.size(), device=self.device) < prob).float()
            action = coef*action + (1-coef)*torch.rand(*action.size(), device=self.device)

            cur_dim = action.size(-1) - self.n_transforms
            max_dim = int((self.max_bins - self.min_bins) * len(self.history) / 100)
            if random.random() < prob and cur_dim < max_dim:
                action = self.preprocess_action(action, normalize=False,
                                                resize=True, des_dim=cur_dim+1)
            new_actions.append(action)
        return new_actions

    def preprocess_action(self, action, normalize=True, resize=False, des_dim=None):
        magnitudes = action[..., :-self.n_transforms]
        probs = action[..., -self.n_transforms:]

        if des_dim is None:
            des_dim = self.max_bins
        if resize:
            magnitudes = linear_resize(magnitudes, des_dim, device=self.device)
        if normalize:
            magnitudes /= magnitudes.sum(-1, keepdim=True)
            probs /= probs.sum(-2, keepdim=True)

        return torch.cat([magnitudes, probs], -1)

    def preprocess_actions(self, actions, *args, **kwargs):
        return torch.stack([self.preprocess_action(action, *args, **kwargs)
                            for action in actions])

    def decode_policy(self, action, *args, **kwargs):
        action = self.preprocess_action(action.detach(), *args, **kwargs)
        return DiscreteRandomApply(self.bag_of_ops, action, self.n_transforms)


class DiscreteRandomApply(torch.nn.Module):
    def __init__(self, bag_of_ops, action, n_transforms):
        super().__init__()
        self.n_transforms = n_transforms
        self.magnitudes = action[..., :-n_transforms]
        self.probs = action[..., -n_transforms:]

        self.layers = [bag_of_ops[i](self.magnitudes[i])
                       for i in range(bag_of_ops.n_ops)]

    def forward(self, image):
        opers = self.probs.multinomial(1).squeeze()
        for i in range(self.n_transforms):
            image = self.layers[opers[i]](image)
        return image


if __name__ == '__main__':
    from discrete_transforms import transforms as bag
    from valuator import Valuator

    '''
    N_TRANSFORMS = 3
    MAX_BINS = 11

    valuator = Valuator([bag.n_ops, N_TRANSFORMS+MAX_BINS])
    agent = EvolveAgent(valuator, bag, n_transforms=N_TRANSFORMS)

    actions = agent.act([0, 0, 0, 0])
    print([action.size() for action in actions])
    policy = agent.decode_policy(actions[0], resize=False)

    agent.update(actions, torch.rand(4))
    actions = agent.act([0] * 8)
    agent.update(actions, torch.rand(8))

    y = torch.rand((32, 3, 32, 32))
    for i in range(3):
        print(policy(y).size())
    '''

    ct = BasicController(bag, n_transforms=4, bins=3)
    print(ct([0] * 8).size())
    ct.reset_bins(5)
    print(ct([0] * 8).size())

