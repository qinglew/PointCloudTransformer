import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Embedding, NeighborEmbedding, OA, SA


class NaivePCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = SA(128)
        self.sa2 = SA(128)
        self.sa3 = SA(128)
        self.sa4 = SA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class SPCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = OA(128)
        self.sa2 = OA(128)
        self.sa3 = OA(128)
        self.sa4 = OA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class PCT(nn.Module):
    def __init__(self, samples=[512, 256]):
        super().__init__()

        self.neighbor_embedding = NeighborEmbedding(samples)
        
        self.oa1 = OA(256)
        self.oa2 = OA(256)
        self.oa3 = OA(256)
        self.oa4 = OA(256)

        self.linear = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.neighbor_embedding(x)

        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class Classification(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Segmentation(nn.Module):
    def __init__(self, part_num):
        super().__init__()

        self.part_num = part_num

        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean, cls_label):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature, cls_label_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x


class NormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 3, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        N = x.size(2)

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)
        
        x = torch.cat([x_max_feature, x_mean_feature, x], dim=1)

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x


"""
Classification networks.
"""

class NaivePCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = NaivePCT()
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class SPCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = SPCT()
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class PCTCls(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.encoder = PCT()
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


"""
Part Segmentation Networks.
"""

class NaivePCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = NaivePCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


class SPCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = SPCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


class PCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = PCT(samples=[1024, 1024])
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


"""
Normal Estimation networks.
"""

class NaivePCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.encoder = NaivePCT()
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


class SPCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.encoder = SPCT()
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


class PCTNormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.encoder = PCT(samples=[1024, 1024])
        self.ne = NormalEstimation()

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.ne(x, x_max, x_mean)
        return x


if __name__ == '__main__':
    pc = torch.rand(4, 3, 1024).to('cuda')
    cls_label = torch.rand(4, 16).to('cuda')

    # testing for cls networks
    naive_pct_cls = NaivePCTCls().to('cuda')
    spct_cls = SPCTCls().to('cuda')
    pct_cls = PCTCls().to('cuda')

    print(naive_pct_cls(pc).size())
    print(spct_cls(pc).size())
    print(pct_cls(pc).size())

    # testing for segmentation networks
    naive_pct_seg = NaivePCTSeg().to('cuda')
    spct_seg = SPCTSeg().to('cuda')
    pct_seg = PCTSeg().to('cuda')

    print(naive_pct_seg(pc, cls_label).size())
    print(spct_seg(pc, cls_label).size())
    print(pct_seg(pc, cls_label).size())

    # testing for normal estimation networks
    naive_pct_ne = NaivePCTNormalEstimation().to('cuda')
    spct_ne = SPCTNormalEstimation().to('cuda')
    pct_ne = PCTNormalEstimation().to('cuda')

    print(naive_pct_ne(pc).size())
    print(spct_ne(pc).size())
    print(pct_ne(pc).size())
