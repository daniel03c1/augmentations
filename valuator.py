import torch
import torch.nn as nn


class Valuator(nn.Module):
    def __init__(self, 
                 input_shape,
                 depth=3,
                 op_bins_dim=16,
                 h_dim=128, 
                 norm='layernorm',
                 **kwargs):
        super(Valuator, self).__init__(**kwargs)

        # [..., op_layers, n_ops, n_bins+1]
        op_layers, n_ops, n_bins = input_shape[-3:]
        self.depth = depth
        if norm == 'layernorm':
            norm_layer = nn.LayerNorm
        elif norm == 'batchnorm':
            norm_layer = nn.BatchNorm
        else:
            raise ValueError(f'unsupported norm type ({norm})')

        self.op_bins_fc = nn.Linear(op_layers*n_bins, op_bins_dim)
        self.op_bins_norm = norm_layer([op_bins_dim])
        self.op_bins_relu = nn.ReLU()

        self.fcs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim)
             if i != 0 else nn.Linear(op_bins_dim*n_ops, h_dim)
             for i in range(depth)])
        self.norms = nn.ModuleList(
            [norm_layer([h_dim]) for i in range(depth)])
        self.relus = nn.ModuleList(
            [nn.ReLU() for i in range(depth)])

        self.final_fc = nn.Linear(h_dim, 1)

    def forward(self, x: torch.Tensor):
        x = x.transpose(-3, -2)
        x = x.reshape(-1, x.size(-3), x.size(-2)*x.size(-1))

        x = self.op_bins_fc(x)
        x = self.op_bins_norm(x)
        x = self.op_bins_relu(x)

        x = x.reshape(-1, x.size(-2) * x.size(-1))
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

