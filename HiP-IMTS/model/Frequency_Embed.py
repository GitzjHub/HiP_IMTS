import torch
import torch.nn as nn

class Frequency_Embedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(Frequency_Embedding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embeddings = nn.Linear(int(seq_len/2)+1, int(self.d_model/2)+1).to(torch.cfloat)

    def forward(self, x_in):
        x = x_in.permute(0,2,1)
        x = torch.fft.rfft(x, dim=-1)
        out = self.embeddings(x)
        x_out = torch.fft.irfft(out, dim=-1, n=self.seq_len)
        x_out = x_out.permute(0, 2, 1)
        return x_out