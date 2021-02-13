import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(self, 
                 emb_size=32, 
                 hidden_size=100, 
                 output_size=16, 
                 n_subpolicies=5,
                 **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, emb_size)
        self.n_subpolicies = n_subpolicies

    def forward(self, x: torch.Tensor):
        # x: [batch, 1] shaped tensor (dtype=torch.long)
        hn_cn = None
        tokens = []

        for i in range(self.n_subpolicies * 4):
            x = self.embedding(x)
            x, hn_cn = self.lstm(x, hn_cn)
            x = self.linear(x) # last layer
            x = x.softmax(-1)

            tokens.append(x)
            x = x.squeeze(1) # [1, 1, dim] to [1, dim]
            x = torch.multinomial(x, 1)
            
        return torch.cat(tokens, axis=1)

