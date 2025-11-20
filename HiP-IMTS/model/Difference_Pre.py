import torch
import torch.nn as nn

class DifferenceDataEmb(nn.Module):
    def __init__(self, hid_dim):
        super(DifferenceDataEmb, self).__init__()
        self.hid_dim = hid_dim
        self.embeddings = nn.Linear(self.hid_dim, self.hid_dim)

    def forward(self, x):
        x_padding = torch.concatenate([x[:, 0:1, :], x], dim=1)
        x_diff = torch.diff(x_padding, dim=1)
        x_diff_emb = self.embeddings(x_diff)

        return x_diff_emb, x_padding

class DataRestoration(nn.Module):
    def __init__(self, hid_dim):
        super(DataRestoration, self).__init__()
        self.hid_dim = hid_dim
        self.projections = nn.Linear(self.hid_dim, self.hid_dim)

    def forward(self, x_diff, x_padding):
        x_diff = self.projections(x_diff)
        x_out = x_diff + x_padding[:, :-1, :]

        return x_out