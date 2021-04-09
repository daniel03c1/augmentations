import math
import torch
import torch.nn as nn
from torchvision import transforms


class SGC(nn.Module):
    """ simple and general controller """
    def __init__(self, 
                 bag_of_ops, 
                 op_layers=4, 
                 n_layers=2, 
                 h_dim=64, 
                 **kwargs):
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
                d_model=h_dim, nhead=4, dim_feedforward=h_dim*2, 
                activation='gelu'),
            num_layers=n_layers)

        # back
        self.mag_mean = nn.Linear(h_dim, self.n_ops) # for op means
        self.mag_std = nn.Linear(h_dim, self.n_ops) # for op stds
        self.prob_mean = nn.Linear(h_dim, self.n_ops) # for mag
        self.prob_std = nn.Linear(h_dim, self.n_ops) # for dist

    def forward(self, x: torch.Tensor, rand_prob=0.):
        n_samples = x.size(0)

        probs = self.input
        probs += self.type_emb(self.type_emb_indices)
        probs += self.order_emb(self.order_emb_indices)

        probs = self.encoder(probs)
        probs = torch.sin(probs) # for test
        probs = probs.repeat(n_samples, 1, 1)

        # action dist
        mag_mean = self.mag_mean(probs[:, :2*self.op_layers]).sigmoid()
        mag_std = self.mag_std(
            probs[:, 2*self.op_layers:4*self.op_layers]).sigmoid()
        prob_mean = self.prob_mean(
            probs[:, -2*self.op_layers:-self.op_layers]).sigmoid()
        prob_std = self.prob_std(probs[:, -self.op_layers:]).sigmoid()

        action_dist = torch.cat([mag_mean, mag_std, prob_mean, prob_std], 1)
        
        # random actions
        rand_action = torch.rand(action_dist.size())
        is_random = torch.randint(2, (n_samples, 1, 1))
        action_dist = (1-is_random)*action_dist + is_random*rand_action

        # actions
        mag = mag_mean \
            + mag_std / 2 * torch.randn((n_samples, *mag_std.size()[1:]))
        prob = prob_mean \
             + prob_std / 2 * torch.randn((n_samples, *prob_std.size()[1:]))

        actions = torch.cat([mag, prob], 1)

        return actions, action_dist

    def calculate_log_probs(self, actions, actions_dist):
        mag, prob = self.split_actions(actions)
        mag_mean, mag_std, prob_mean, prob_std = self.split_actions_dist(
            actions_dist)

        mag_log_prob = - ((mag - mag_mean)**2) / (2 * ((mag_std/2)**2)) \
                       - mag_std.log() - math.log(math.sqrt(2*math.pi))
        prob_log_prob = - ((prob - prob_mean)**2) / (2 * ((prob_std/2)**2)) \
                        - prob_std.log() - math.log(math.sqrt(2*math.pi))
        
        return torch.cat([mag_log_prob, prob_log_prob], 1)

    def calculate_entropy(self, actions_dist):
        # TODO: need to exactly calculate entropy
        mag_mean, mag_std, prob_mean, prob_std = self.split_actions_dist(
            actions_dist)

        ent_const = 0.5 + 0.5 * math.log(2*math.pi)

        mag_ent = ent_const + mag_std.log()
        prob_ent = ent_const + prob_std.log()
        dist_ent = prob_mean / prob_mean.sum(-1, keepdims=True)
        dist_ent = -(dist_ent*dist_ent.log())
        dist_prob_ent = ent_const + prob_std.log()

        return torch.cat([mag_ent, prob_ent, dist_ent, dist_prob_ent], 1).mean()

    def decode_policy(self, action):
        assert action.ndim == 2
        action = action.clip(0, 1)
        mag = action[:-self.op_layers]
        prob = action[-self.op_layers:]

        prob = prob / prob.sum(-1, keepdims=True)

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


class SGCv2(nn.Module):
    """ simple and general controller """
    def __init__(self, 
                 bag_of_ops, 
                 op_layers=4, 
                 n_layers=2, 
                 h_dim=64, 
                 **kwargs):
        super(SGCv2, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.op_layers = op_layers
        self.h_dim = h_dim

        ''' modules '''
        # fixed inputs
        # [op_layer*6, n_ops, h_dim]
        seq_len = op_layers * 6 # [mean, std] of mag*2, [mean, std] of prob

        self.op_emb_indices = torch.arange(self.n_ops).reshape(1, 1, -1)
        self.type_emb_indices = torch.arange(6) \
                                     .repeat_interleave(op_layers) \
                                     .reshape(1, -1, 1)
        self.order_emb_indices = torch.arange(op_layers).repeat(6) \
                                      .reshape(1, -1, 1)

        # front
        # [mag_mean, mag_std, prob_mean, prob_std]
        self.op_emb = nn.Embedding(self.n_ops, h_dim) 
        self.type_emb = nn.Embedding(6, h_dim) 
        self.order_emb = nn.Embedding(op_layers, h_dim) # [1, 2, ..., op_layer]

        # middle
        self.n_layers = n_layers
        self.fcs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for i in range(n_layers*2)])
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm([seq_len, self.n_ops, h_dim]) 
             for i in range(n_layers*2)])

        # back
        self.to_score = nn.Linear(h_dim, 1)

    def forward(self, x: torch.Tensor, rand_prob=0.):
        n_samples = x.size(0)

        op_emb = self.op_emb(self.op_emb_indices)
        type_emb = self.type_emb(self.type_emb_indices)
        order_emb = self.order_emb(self.order_emb_indices)

        # [op_layer*6, n_ops, h_dim]
        emb = op_emb + type_emb + order_emb
        probs = 0
        for i in range(self.n_layers):
            # similar to simple conv block
            probs = emb + probs
            org = probs

            probs = self.fcs[i*2](probs)
            probs = self.layernorms[i*2](probs)
            probs = probs * probs.sigmoid() # swish

            probs = self.fcs[i*2+1](probs)
            probs = self.layernorms[i*2+1](probs)
            probs = probs + org
            probs = probs * probs.sigmoid() # swish

        probs = self.to_score(probs).sigmoid()
        probs = probs.squeeze(-1)

        probs = probs.repeat(n_samples, 1, 1)

        # action dist
        mag_mean = probs[:, :2*self.op_layers]
        mag_std = probs[:, 2*self.op_layers:4*self.op_layers]
        prob_mean = probs[:, -2*self.op_layers:-self.op_layers]
        prob_std = probs[:, -self.op_layers:]

        action_dist = torch.cat([mag_mean, mag_std, prob_mean, prob_std], 1)
        
        # random actions
        rand_action = torch.rand(action_dist.size())
        is_random = torch.randint(2, (n_samples, 1, 1))
        action_dist = (1-is_random)*action_dist + is_random*rand_action

        # actions
        mag = mag_mean \
            + mag_std / 2 * torch.randn((n_samples, *mag_std.size()[1:]))
        prob = prob_mean \
             + prob_std / 2 * torch.randn((n_samples, *prob_std.size()[1:]))

        actions = torch.cat([mag, prob], 1)

        return actions, action_dist

    def calculate_log_probs(self, actions, actions_dist):
        mag, prob = self.split_actions(actions)
        mag_mean, mag_std, prob_mean, prob_std = self.split_actions_dist(
            actions_dist)

        mag_log_prob = - ((mag - mag_mean)**2) / (2 * ((mag_std/2)**2)) \
                       - mag_std.log() - math.log(math.sqrt(2*math.pi))
        prob_log_prob = - ((prob - prob_mean)**2) / (2 * ((prob_std/2)**2)) \
                        - prob_std.log() - math.log(math.sqrt(2*math.pi))
        
        return torch.cat([mag_log_prob, prob_log_prob], 1)

    def calculate_entropy(self, actions_dist):
        # TODO: need to exactly calculate entropy
        mag_mean, mag_std, prob_mean, prob_std = self.split_actions_dist(
            actions_dist)

        ent_const = 0.5 + 0.5 * math.log(2*math.pi)

        mag_ent = ent_const + mag_std.log()
        prob_ent = ent_const + prob_std.log()
        dist_ent = prob_mean / prob_mean.sum(-1, keepdims=True)
        dist_ent = -(dist_ent*dist_ent.log())
        dist_prob_ent = ent_const + prob_std.log()

        return torch.cat([mag_ent, prob_ent, dist_ent, dist_prob_ent], 1).mean()

    def decode_policy(self, action):
        assert action.ndim == 2
        action = action.clip(0, 1)
        mag = action[:-self.op_layers]
        prob = action[-self.op_layers:]

        prob = prob / prob.sum(-1, keepdims=True)

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


class SGCvD(nn.Module):
    """ simple and general controller (discrete version) """
    def __init__(self, 
                 bag_of_ops, 
                 op_layers=4, 
                 n_layers=3,
                 h_dim=64, 
                 dropout_p=0.3,
                 bins=10,
                 **kwargs):
        super(SGCvD, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.op_layers = op_layers
        self.h_dim = h_dim

        ''' modules '''
        # fixed inputs
        # [op_layer, n_ops, 10 + 1]
        self.op_emb_indices = torch.arange(self.n_ops).reshape(1, 1, -1)
        self.order_emb_indices = torch.arange(op_layers).reshape(1, -1, 1)

        # front
        self.op_emb = nn.Embedding(self.n_ops, h_dim) 
        self.op_emb_do = nn.Dropout(dropout_p)
        self.order_emb = nn.Embedding(op_layers, h_dim)
        self.order_emb_do = nn.Dropout(dropout_p)

        # middle
        self.n_layers = n_layers
        self.fcs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for i in range(n_layers*2)])
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm([op_layers, self.n_ops, h_dim]) 
             for i in range(n_layers*2)])
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm([op_layers, self.n_ops, h_dim]) 
             for i in range(n_layers*2)])
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_p) for i in range(n_layers)])

        # back
        self.last_layernorm = nn.LayerNorm([op_layers, self.n_ops, h_dim])
        self.to_magnitude = nn.Linear(h_dim, bins)
        self.to_prob = nn.Linear(h_dim*self.n_ops, self.n_ops)

    def forward(self, x: torch.Tensor, rand_prob=0.):
        n_samples = x.size(0)

        op_emb = self.op_emb(self.op_emb_indices)
        order_emb = self.order_emb(self.order_emb_indices)

        # [batch, op_layer, n_ops, h_dim]
        x = self.op_emb_do(op_emb) + self.order_emb_do(order_emb)
        x = x.repeat(n_samples, 1, 1, 1)

        for i in range(self.n_layers):
            # similar to simple conv block
            org = x

            x = self.fcs[i*2](x)
            x = self.layernorms[i*2](x)
            x = x * x.sigmoid() # swish

            x = self.dropouts[i](x)

            x = self.fcs[i*2+1](x)
            x = self.layernorms[i*2+1](x)
            x = x + org
            x = x * x.sigmoid() # swish

        x = self.last_layernorm(x)

        # [batch, op_layer, n_ops, bins]
        magnitudes = self.to_magnitude(x)
        magnitudes = nn.functional.softmax(magnitudes, dim=-1)

        is_rand = torch.randint(2, (n_samples, 1, 1, 1))
        rand_mag = torch.rand(magnitudes.size())
        rand_mag /= rand_mag.sum(-1, keepdim=True)
        magnitudes = (1-is_rand)*magnitudes + is_rand*rand_mag

        # [batch, op_layer, n_ops, 1]
        probs = x.view(-1, self.op_layers, self.n_ops*self.h_dim)
        probs = self.to_prob(probs)
        probs = nn.functional.softmax(probs, dim=-1)

        is_rand = is_rand.squeeze(-1)
        rand_prob = torch.rand(probs.size())
        rand_prob /= rand_prob.sum(-1, keepdim=True)
        probs = (1-is_rand)*probs + is_rand*rand_prob

        print(probs.size(), magnitudes.size())

        return torch.cat([magnitudes, probs.unsqueeze(-1)], -1)

    def calculate_log_probs(self, actions):
        return actions.log()

    def calculate_entropy(self, actions):
        return 0

    def decode_policy(self, action):
        return RandomApply(self.bag_of_ops, prob, mag, double_mag=True)


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


class RandomApplyDiscrete(torch.nn.Module):
    def __init__(self, controller_action):
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

    brain = SGCvD(bag, op_layers=3)
    print(sum([p.numel() for p in brain.parameters()]))

    x = torch.zeros(2)
    actions = brain(x, rand_prob=0.)
    print(actions) # .size())

    '''
    log_prob = brain.calculate_log_probs(action, prob)
    entropy = brain.calculate_entropy(prob)

    policy = brain.decode_policy(action[0])
    y = torch.rand((32, 3, 32, 32))
    for i in range(5):
        print(policy(y).size())
    '''

