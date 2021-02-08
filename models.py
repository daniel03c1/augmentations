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
        self.softmax = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(output_size, emb_size)
        self.n_subpolicies = n_subpolicies

    def forward(self):
        x = torch.zeros((1, 1), dtype=torch.long)
        hn_cn = None

        tokens = []

        for i in range(self.n_subpolicies * 4):
            x = self.embedding(x)
            x, hn_cn = self.lstm(x, hn_cn)
            x = self.softmax(x)

            tokens.append(x)
            x = torch.argmax(x, -1)
            
        return torch.cat(tokens, axis=0).squeeze()

