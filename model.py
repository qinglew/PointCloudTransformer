import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import NeighborEmbedding, SA, ChannelAttention, Classification, MultiLevelAttention, DoubleAttention


class Baseline1(nn.Module):
    """
    Shared MLP + NeighborEmbedding(512, 256) + MaxPooling
    """
    def __init__(self):
        super(Baseline1, self).__init__()

        self.embedding = NeighborEmbedding()
        self.decoder = Classification()
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.max(x, dim=-1)[0]
        x = self.decoder(x)
        return x


class Baseline2(nn.Module):
    """
    Shared MLP + NeighborEmbedding(512, 256) + SA + MaxPooling
    """
    def __init__(self):
        super(Baseline2, self).__init__()

        self.embedding = NeighborEmbedding()
        self.sa = SA(512)
        self.decoder = Classification()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.sa(x)
        x = torch.max(x, dim=-1)[0]
        x = self.decoder(x)
        return x


class MLA1(nn.Module):
    """
    Shared MLP + NeighborEmbedding(512, 256) + (SA + CA) + MaxPooling  91.45
    """
    def __init__(self):
        super(MLA1, self).__init__()
        
        self.embedding = NeighborEmbedding()
        self.mla = MultiLevelAttention()
        self.decoder = Classification()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.mla(x)
        x = torch.max(x, dim=-1)[0]
        x = self.decoder(x)
        return x


class MLA2(nn.Module):
    """
    Shared MLP + NeighborEmbedding(512, 256) + (SA + DoubleAttention) + MaxPooling
    """
    def __init__(self):
        super(MLA2, self).__init__()
        
        self.embedding = NeighborEmbedding()
        self.sa = SA(512)
        self.da = DoubleAttention(512)
        self.decoder = Classification()
    
    def forward(self, x):
        x = self.embedding(x)
        x1 = self.sa(x)
        x2 = self.da(x)
        x = x + x1 + x2
        x = torch.max(x, dim=-1)[0]
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    pc = torch.rand(4, 3, 1024).to('cuda')
    mla = MLA1().to('cuda')
    out = mla(pc)
    print(out.size())
