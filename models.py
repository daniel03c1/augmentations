import math
import torch
import torch.nn as nn
from torchvision import transforms


class Controller(nn.Module):
    def __init__(self, 
                 bag_of_ops,
                 emb_size=32, 
                 hidden_size=100, 
                 n_subpolicies=5,
                 **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        output_size = bag_of_ops.n_ops

        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        # self.linear = nn.Linear(hidden_size, output_size)
        # self.embedding = nn.Embedding(output_size, emb_size)
        self.mag_linear = nn.Linear(hidden_size, output_size)
        self.mag_emb = nn.Embedding(output_size, emb_size)
        self.op_linear = nn.Linear(hidden_size, output_size)
        self.op_emb = nn.Embedding(output_size, emb_size)
        self.n_subpolicies = n_subpolicies

    def forward(self, x: torch.Tensor):
        # x: [batch, 1] shaped tensor (dtype=torch.long)
        org_x = x
        hn_cn = None
        tokens = []

        for i in range(self.n_subpolicies * 4):
            is_op = i % 2 == 0
            if is_op:
                x = self.mag_emb(x)
            else:
                x = self.op_emb(x)
            # x = self.embedding(x)
            x, hn_cn = self.lstm(x, hn_cn)

            if is_op:
                x = self.op_linear(x)
            else:
                x = self.mag_linear(x)
            # x = self.linear(x) # last layer
            x = x.softmax(-1)

            tokens.append(x)
            x = x.squeeze(1) # [1, 1, dim] to [1, dim]
            x = torch.multinomial(x, 1)
            
        return torch.cat(tokens, axis=1)

    def decode_policy(self, policy):
        n_subpolicies = policy.size(-2) // 4
        n_magnitudes = policy.size(-1)

        probs = policy
        policy = torch.multinomial(policy, 1).squeeze(-1)
        policy = [policy[i*4:(i+1)*4]
                  for i in range(n_subpolicies)]

        policy = [
            transforms.Compose([
                self.bag_of_ops[op1]((mag1+1)/n_magnitudes),
                self.bag_of_ops[op2]((mag2+1)/n_magnitudes)
            ]) 
            for op1, mag1, op2, mag2 in policy]

        return transforms.RandomChoice(policy)

    def entropy(self, probs):
        return (probs * probs.log()).sum(-1)


class PolicyController(nn.Module):
    def __init__(self, bag_of_ops, op_layers=4, n_layers=2, h_dim=64, **kwargs):
        super(PolicyController, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.op_layers = op_layers
        self.h_dim = h_dim
        self.max_ratio = 3

        ''' modules '''
        # fixed inputs
        seq_len = 2 + op_layers * 2
        self.input = torch.normal(
            torch.zeros((1, seq_len, h_dim)), torch.ones(1, seq_len, h_dim))
        self.type_emb_indices = torch.cat(
            [torch.zeros((1, 2)),
             torch.ones((1, op_layers)),
             2*torch.ones((1, op_layers))],
            -1).long()
        self.order_emb_indices = torch.arange(op_layers).repeat(2).reshape(1, -1)

        # front
        self.type_emb = nn.Embedding(3, h_dim) # [op, mag, dist]
        self.order_emb = nn.Embedding(op_layers, h_dim) # [1, 2,...]

        # middle
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim, nhead=4, dim_feedforward=h_dim*2),
            num_layers=n_layers)

        # back
        self.mean_linear = nn.Linear(h_dim, self.n_ops) # for op means
        self.std_linear = nn.Linear(h_dim, self.n_ops) # for op stds
        self.mag_linear = nn.Linear(h_dim, self.n_ops) # for mag
        self.dist_linear = nn.Linear(h_dim, self.n_ops) # for dist

    def forward(self, x: torch.Tensor):
        probs = self.input
        probs += self.type_emb(self.type_emb_indices)
        probs += torch.cat(
            [torch.zeros((1, 2, self.h_dim)),
             self.order_emb(self.order_emb_indices)],
            axis=1)

        probs = self.encoder(probs)

        mean = self.mean_linear(probs[:, :1]).sigmoid() # [0, 1]
        std = self.std_linear(probs[:, 1:2]).sigmoid() / 4 # [0, 0.25]
        mag = self.mag_linear(probs[:, 2:-self.op_layers]).softmax(-1)
        dist = self.mag_linear(probs[:, -self.op_layers:])
        dist = torch.sigmoid(dist) + 1/(self.max_ratio-1)
        dist /= dist.sum(-1, keepdims=True)

        action_dist = torch.cat([mean, std, mag, dist], 1)

        # act
        n_samples = x.size(0)
        max_mags = mean + torch.randn((n_samples, 1, self.n_ops)) * std
        max_mags = torch.clip(max_mags, 0, 1)
        mag_per_layer = torch.multinomial(mag[0], n_samples, True)
        mag_per_layer = mag_per_layer.transpose(1, 0)
        mag_per_layer = torch.eye(self.n_ops)[mag_per_layer]

        actions = torch.cat(
            [max_mags, mag_per_layer, dist.repeat(n_samples, 1, 1)] , 1)
        action_dist = action_dist.repeat(n_samples, 1, 1)

        return actions, action_dist

    def calculate_log_probs(self, actions, actions_dist):
        # [n_samples, n_ops]
        max_mags = actions[:, 0]
        # [n_samples, op_layers, n_ops]
        mag_per_layer = actions[:, 1:self.op_layers+1] 
        # [n_samples, op_layers, n_ops]
        op_dist = actions[:, -self.op_layers:]

        mean = actions_dist[:, 0] # [n_samples, n_ops]
        std = actions_dist[:, 1] # [n_samples, n_ops]
        mag = actions_dist[:, 2:-self.op_layers]
        dist = actions_dist[:, -self.op_layers:]

        # [n_samples, n_ops]
        op_log_prob = -((max_mags - mean) ** 2) \
            / (2 * (std**2)) - std.log() - math.log(math.sqrt(2 * math.pi))
        # [n_samples, op_layers]
        mag_log_prob = (mag_per_layer * mag).sum(-1).log()
        # [n_samples, op_layers * n_ops]
        dist_log_prob = dist.log().view(dist.size(0), -1)
        
        return torch.cat([op_log_prob, mag_log_prob, dist_log_prob], 1)

    def calculate_entropy(self, actions_dist):
        # TODO: need to exactly calculate entropy
        std = actions_dist[:, 1] # [n_samples, n_ops]
        rest = actions_dist[:, 2:] 

        std_ent = 0.5 + 0.5 * math.log(2 * math.pi) + std.log()
        rest_ent = (rest*rest.log()).sum(-1) # [n_samples, op_layers*2]
        
        return torch.cat([std_ent, rest_ent], 1).mean()

    def decode_policy(self, action):
        assert action.ndim == 2
        max_mags = action[0:1] # [1, n_ops]
        mag_per_layer = action[1:self.op_layers+1] # [op_layers, n_ops]
        op_dist = action[-self.op_layers:]

        # from categorial to real num
        mag_per_layer = (mag_per_layer.argmax(-1, keepdims=True)+1) \
                      / self.n_ops
        magnitudes = max_mags * mag_per_layer

        return RandomApply(self.bag_of_ops, op_dist, magnitudes)


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
        seq_len = op_layers * 4 # [mean, std] of mag, [mean, std] of prob
        self.input = torch.normal(
            torch.zeros((1, seq_len, h_dim)), torch.ones(1, seq_len, h_dim))
        self.type_emb_indices = torch.arange(4) \
                                     .repeat_interleave(op_layers) \
                                     .unsqueeze(0)
        self.order_emb_indices = torch.arange(op_layers).repeat(4).unsqueeze(0)

        # front
        # [mag_mean, mag_std, prob_mean, prob_std]
        self.type_emb = nn.Embedding(4, h_dim) 
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
        mag_mean = self.mag_mean(probs[:, :self.op_layers]).sigmoid()
        mag_std = self.mag_std(
            probs[:, self.op_layers:2*self.op_layers]).sigmoid()
        prob_mean = self.prob_mean(
            probs[:, 2*self.op_layers:3*self.op_layers]).sigmoid()
        prob_std = self.prob_std(
            probs[:, 3*self.op_layers:4*self.op_layers]).sigmoid()

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
        
        return mag_log_prob + prob_log_prob

    def calculate_entropy(self, actions_dist):
        # TODO: need to exactly calculate entropy
        mag_mean, mag_std, prob_mean, prob_std = self.split_actions_dist(
            actions_dist)

        mag_ent = 0.5 + 0.5 * math.log(2*math.pi) + mag_std.log()
        prob_ent = 0.5 + 0.5 * math.log(2*math.pi) + prob_std.log()
        dist_ent = prob_mean.softmax(-1)
        dist_ent = (dist_ent*dist_ent.log())

        return mag_ent.mean() + prob_ent.mean() + dist_ent.mean()

    def decode_policy(self, action):
        assert action.ndim == 2
        mag = action[:self.op_layers]
        prob = action[-self.op_layers:]

        mag = mag.clip(0, 1)
        prob = prob.softmax(-1)

        return RandomApply(self.bag_of_ops, prob, mag)

    def split_actions(self, actions):
        mag = actions[:, :self.op_layers]
        prob = actions[:, -self.op_layers:]
        return mag, prob

    def split_actions_dist(self, actions_dist):
        mag_mean = actions_dist[:, :self.op_layers]
        mag_std = actions_dist[:, 1*self.op_layers:2*self.op_layers]
        prob_mean = actions_dist[:, 2*self.op_layers:3*self.op_layers]
        prob_std = actions_dist[:, 3*self.op_layers:4*self.op_layers]
        return mag_mean, mag_std, prob_mean, prob_std


class RandomApply(torch.nn.Module):
    def __init__(self, 
                 bag_of_ops, 
                 probs_per_layer, # [op_layers, n_ops]
                 magnitudes): # [op_layers, n_ops]
        super().__init__()
        self.probs_per_layer = probs_per_layer
        self.op_layers = magnitudes.size(0)

        self.layers = [
            [bag_of_ops[i](magnitudes[j, i].cpu().item()) 
             for i in range(bag_of_ops.n_ops)]
            for j in range(self.op_layers)]

    #torch.jit.script
    def forward(self, image):
        opers = self.probs_per_layer.multinomial(1).squeeze()
        for i in range(self.op_layers):
            image = self.layers[i][opers[i]](image)
        return image


if __name__ == '__main__':
    from transforms import transforms
    from utils import get_default_device

    bag = transforms

    c = Controller(bag)
    print(sum([p.numel() for p in c.parameters()]))

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

