import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import NeighborEmbedding, SA, ChannelAttention, Classification, MultiLevelAttention


class MLA(nn.Module):
    def __init__(self):
        super(MLA, self).__init__()
        
        self.embedding = NeighborEmbedding()
        self.mla = MultiLevelAttention()
        self.decoder = Classification()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.mla(x)
        x = torch.max(x, dim=-1)[0]
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    pc = torch.rand(4, 3, 1024).to('cuda')
    mla = MLA().to('cuda')
    out = mla(pc)
    print(out.size())
