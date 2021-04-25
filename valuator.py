import torch
import torch.nn as nn


class Valuator(nn.Module):
    def __init__(self, 
                 input_shape,
                 depth=3,
                 h_dim=128,
                 **kwargs):
        super(Valuator, self).__init__(**kwargs)

        n_ops, dims = input_shape[-2:]
        self.depth = depth

        self.op_bins_fc = nn.Linear(n_ops * dims, h_dim)
        self.op_bins_norm = nn.BatchNorm1d(h_dim)
        self.op_bins_relu = nn.ReLU()

        self.fcs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for i in range(depth)])
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(h_dim) for i in range(depth)])
        self.relus = nn.ModuleList(
            [nn.ReLU() for i in range(depth)])

        self.final_fc = nn.Linear(h_dim, 1)

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, x.size(-2)*x.size(-1))

        x = self.op_bins_fc(x)
        x = self.op_bins_norm(x)
        x = self.op_bins_relu(x)

        for i in range(self.depth):
            x = self.fcs[i](x)
            x = self.norms[i](x)
            x = self.relus[i](x)

        x = self.final_fc(x)
        return x


if __name__ == '__main__':
    input_shape = [2, 17, 22]
    brain = Valuator(input_shape)
    print(sum([p.numel() for p in brain.parameters()]))

    x = torch.randn([4]+input_shape)
    y = brain(x)
    print(y)
