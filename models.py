import math
import torch
import torch.nn as nn
from torchvision import transforms


class SGC(nn.Module):
    """ simple and general controller """
    def __init__(self, 
                 bag_of_ops, 
                 op_layers=4, 
                 n_layers=2, h_dim=64, **kwargs):
        super(SGC, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.op_layers = op_layers
        self.h_dim = h_dim

        ''' modules '''
        # fixed inputs
        seq_len = op_layers * 6 # [mean, std] of mag*2, [mean, std] of prob
        self.input = torch.normal(
            torch.zeros((1, seq_len, h_dim)), torch.ones(1, seq_len, h_dim))
        self.type_emb_indices = torch.arange(6) \
                                     .repeat_interleave(op_layers) \
                                     .unsqueeze(0)
        self.order_emb_indices = torch.arange(op_layers).repeat(6).unsqueeze(0)

        # front
        # [mag_mean, mag_std, prob_mean, prob_std]
        self.type_emb = nn.Embedding(6, h_dim) 
        self.order_emb = nn.Embedding(op_layers, h_dim) # [1, 2, ..., op_layer]

        # middle
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim, nhead=4, dim_feedforward=h_dim*2),
            num_layers=n_layers)

        # back
        self.mag_mean = nn.Linear(h_dim, self.n_ops) # for op means
        self.mag_std = nn.Linear(h_dim, self.n_ops) # for op stds
        self.prob_mean = nn.Linear(h_dim, self.n_ops) # for mag
        self.prob_std = nn.Linear(h_dim, self.n_ops) # for dist

    def forward(self, x: torch.Tensor):
        probs = self.input
        probs += self.type_emb(self.type_emb_indices)
        probs += self.order_emb(self.order_emb_indices)

        probs = self.encoder(probs)

        # action dist
        mag_mean = self.mag_mean(probs[:, :2*self.op_layers]).sigmoid()
        mag_std = self.mag_std(
            probs[:, 2*self.op_layers:4*self.op_layers]).sigmoid()
        prob_mean = self.prob_mean(
            probs[:, -2*self.op_layers:-self.op_layers]).sigmoid()
        prob_std = self.prob_std(probs[:, -self.op_layers:]).sigmoid()

        action_dist = torch.cat([mag_mean, mag_std, prob_mean, prob_std], 1)

        # actions
        n_samples = x.size(0)
        mag = mag_mean \
            + mag_std / 2 * torch.randn((n_samples, *mag_std.size()[1:]))
        prob = prob_mean \
             + prob_std / 2 * torch.randn((n_samples, *prob_std.size()[1:]))

        actions = torch.cat([mag, prob], 1)
        action_dist = action_dist.repeat(n_samples, 1, 1)

        return actions, action_dist

    def calculate_log_probs(self, actions, actions_dist):
        mag, prob = self.split_actions(actions)
        mag_mean, mag_std, prob_mean, prob_std = self.split_actions_dist(
            actions_dist)

        mag_log_prob = - ((mag - mag_mean)**2) / (2 * (mag_std**2)) \
                       - mag_std.log() - math.log(math.sqrt(2*math.pi))
        prob_log_prob = - ((prob - prob_mean)**2) / (2 * (prob_std**2)) \
                        - prob_std.log() - math.log(math.sqrt(2*math.pi))
        
        return torch.cat([mag_log_prob, prob_log_prob], 1)

    def calculate_entropy(self, actions_dist):
        # TODO: need to exactly calculate entropy
        mag_mean, mag_std, prob_mean, prob_std = self.split_actions_dist(
            actions_dist)

        mag_ent = 0.5 + 0.5 * math.log(2*math.pi) + mag_std.log()
        prob_ent = 0.5 + 0.5 * math.log(2*math.pi) + prob_std.log()
        dist_ent = prob_mean.softmax(-1)
        dist_ent = -(dist_ent*dist_ent.log())

        return torch.cat([mag_ent, prob_ent, dist_ent], 1).mean()

    def decode_policy(self, action):
        assert action.ndim == 2
        mag = action[:-self.op_layers]
        prob = action[-self.op_layers:]

        mag = mag.clip(0, 1)
        prob = prob.softmax(-1)

        return RandomApply(self.bag_of_ops, prob, mag, double_mag=True)

    def split_actions(self, actions):
        mag = actions[:, :-self.op_layers]
        prob = actions[:, -self.op_layers:]
        return mag, prob

    def split_actions_dist(self, actions_dist):
        mag_mean = actions_dist[:, :2*self.op_layers]
        mag_std = actions_dist[:, 2*self.op_layers:4*self.op_layers]
        prob_mean = actions_dist[:, -2*self.op_layers:-self.op_layers]
        prob_std = actions_dist[:, -self.op_layers:]
        return mag_mean, mag_std, prob_mean, prob_std


class RandomApply(torch.nn.Module):
    def __init__(self, 
                 bag_of_ops, 
                 probs_per_layer, # [op_layers, n_ops]
                 magnitudes, # [op_layers, n_ops]
                 double_mag=False):
        super().__init__()
        self.probs_per_layer = probs_per_layer
        self.op_layers = magnitudes.size(0)
        if double_mag:
            self.op_layers = self.op_layers // 2

        self.layers = [
            [bag_of_ops[i]([magnitudes[j, i].cpu().item(), 
                            magnitudes[j+self.op_layers, i].cpu().item()]) 
             for i in range(bag_of_ops.n_ops)]
            for j in range(self.op_layers)]

    # @torch.jit.script
    def forward(self, image):
        opers = self.probs_per_layer.multinomial(1).squeeze()
        for i in range(self.op_layers):
            image = self.layers[i][opers[i]](image)
        return image


if __name__ == '__main__':
    from transforms import transforms
    from utils import get_default_device

    bag = transforms

    brain = SGC(bag, op_layers=4) # PolicyController(bag, op_layers=4)
    print(sum([p.numel() for p in brain.parameters()]))

    x = torch.zeros(2)
    action, prob = brain(x)
    log_prob = brain.calculate_log_probs(action, prob)
    entropy = brain.calculate_entropy(prob)

    print('action', action)
    print('prob', prob)
    print('log_prob', log_prob)
    print('entropy', entropy)

    policy = brain.decode_policy(action[0])
    y = torch.rand((32, 3, 32, 32))
    for i in range(5):
        print(policy(y).size())

