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


class ContBrain(nn.Module):
    def __init__(self, bag_of_ops, op_layers=4, n_layers=1, h_dim=64, **kwargs):
        super(ContBrain, self).__init__(**kwargs)
        self.bag_of_ops = bag_of_ops
        self.n_ops = bag_of_ops.n_ops
        self.op_layers = op_layers
        self.h_dim = h_dim
        self.max_ratio = 3

        # modules
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim, nhead=4, dim_feedforward=h_dim*4),
            num_layers=n_layers)
        self.embedding = nn.Embedding(3, h_dim) # [op, mag, dist]
        self.op_linear = nn.Linear(h_dim, self.n_ops) # for op
        self.mag_linear = nn.Linear(h_dim, self.n_ops) # for mag
        self.dist_linear = nn.Linear(h_dim, self.n_ops) # for dist

        # inputs
        seq_len = bag_of_ops.n_ops + op_layers * 2
        self.input = torch.normal(
            mean=torch.zeros((1, seq_len, h_dim)),
            std=torch.ones(1, seq_len, h_dim))
        self.emb_indices = torch.cat(
            [torch.zeros((1, self.n_ops)),
             torch.ones((1, op_layers)),
             2*torch.ones((1, op_layers))],
            -1).long()

        self.scale = torch.arange(0, 1+1e-8, 1/(self.n_ops-1))

    def forward(self, x: torch.Tensor):
        out = self.input + self.embedding(self.emb_indices) + x
        out = self.encoder(out)

        op = self.op_linear(out[:, :self.n_ops]).softmax(-1)
        mag = self.mag_linear(out[:, self.n_ops:-self.op_layers]).softmax(-1)
        dist = self.dist_linear(out[:, -self.op_layers:])
        dist = torch.sigmoid(dist) + 1/(self.max_ratio-1)
        dist /= dist.sum(-1, keepdims=True)
        out = torch.cat([op, mag, dist], 1)

        return out

    def decode_policy(self, policy):
        assert policy.ndim == 2
        # map categorical dist to magnitudes in range of [0, 1]
        magnitudes_per_op = torch.matmul(policy[:self.n_ops], self.scale)
        magnitudes_per_op = torch.clip(magnitudes_per_op, 0, 1)
        magnitudes_per_layer = torch.matmul(
            policy[self.n_ops:-self.op_layers], self.scale)
        magnitudes_per_layer = torch.clip(magnitudes_per_layer, 0, 1)

        probs_per_layer = policy[-self.op_layers:]

        return RandomApply(self.bag_of_ops, 
                           probs_per_layer, 
                           magnitudes_per_op,
                           magnitudes_per_layer)


class RandomApply(torch.nn.Module):
    def __init__(self, 
                 bag_of_ops, 
                 probs_per_layer, # [op_layers, n_ops]
                 magnitudes_per_op, # [n_ops]
                 magnitudes_per_layer):
        super().__init__()
        self.probs_per_layer = probs_per_layer
        self.op_layers = magnitudes_per_layer.size(0)

        magnitudes = magnitudes_per_layer.unsqueeze(1) \
                   * magnitudes_per_op.unsqueeze(0)
        magnitudes = torch.clip(magnitudes, 0, 1)

        self.layers = [
            [bag_of_ops[i](magnitudes[j, i].cpu().item()) 
             for i in range(bag_of_ops.n_ops)]
            for j in range(self.op_layers)]

    def forward(self, image):
        for i in range(self.op_layers):
            op = self.probs_per_layer[i].multinomial(1)[0]
            image = self.layers[i][op](image)
        return image


if __name__ == '__main__':
    from transforms import transforms
    from utils import get_default_device

    bag = transforms

    c = Controller(bag)
    print(sum([p.numel() for p in c.parameters()]))

    brain = ContBrain(bag, op_layers=4)
    brain.load_state_dict(torch.load('WR_contbrain_ppo.pt'))
    print(sum([p.numel() for p in brain.parameters()]))

    x = torch.normal(torch.zeros((4, 1, 64)), torch.ones((4, 1, 64)))
    out = brain(x)
    import pdb; pdb.set_trace()
    policy = brain.decode_policy(out[0])
    y = torch.rand((32, 3, 32, 32))
    for i in range(5):
        print(policy(y).size())

