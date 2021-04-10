import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import sample_and_knn_group


class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        new_xyz, new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords, features=x)  # [B, s, 3], [B, s, k, in_channels]
        new_feature = new_feature.permute(0, 3, 1, 2)                               # [B, in_channels, s, k]
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                     # [B, out_channels, s, k]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                     # [B, out_channels, s, k]
        new_feature = torch.max(new_feature, dim=-1)[0]                             # [B, out_channels, s]
        return new_xyz, new_feature


class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[512, 256]):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
        self.sg2 = SG(s=samples[1], in_channels=256, out_channels=512)
    
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        xyz1, features1 = self.sg1(features, xyz)         # [B, 128, 512]
        _, features2 = self.sg2(features1, xyz1)          # [B, 512, 256]

        return features2


class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # residual
        x = x + x_s

        return x


class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)   # [B, N, d_k]
        x_k = self.k_conv(x)                    # [B, d_k, N]
        x_v = self.v_conv(x)                    # [B, de, N]

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here [B, N, N]

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention module for channel-wise releationship.
    """

    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

        # self.linear_trans = nn.Sequential(
        #     nn.Conv1d(channels, channels, 1),
        #     nn.BatchNorm1d(channels),
        #     nn.ReLU()
        # )
    
    def forward(self, x):
        """
        Input:
            x: [B, d_in, N]
        
        Output:
            x: [B, d_in, N]
        """
        x_t = x.permute(0, 2, 1)    # [B, N, d_in]

        # compute attention map
        energy = torch.bmm(x, x_t)  # [B, d_in, d_in]
        attention = self.softmax(energy)  # [B, d_in, d_in]

        # weighted sum
        x_s = torch.bmm(x_t, attention)  # [B, N, d_in]

        x_s = x_s.permute(0, 2, 1)  # [B, d_in, N]
        # TODO: use a linear transformation
        # x_s = self.linear_trans(x_s)

        # residual
        x = x + x_s

        return x


class MultiLevelAttention(nn.Module):
    def __init__(self):
        super(MultiLevelAttention, self).__init__()
        self.sa = SA(512)
        self.ca = ChannelAttention(512)
        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x1 = self.sa(x)
        x2 = self.ca(x)
        x = x1 + x2
        x = self.linear(x)
        return x


class AttentionAggregation(nn.Module):
    def __init__(self, d_in, d_k, d_v, d_g):
        super(AttentionAggregation, self).__init__()
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        
        self.q_conv = nn.Conv1d(d_in, d_k, 1, bias=False)
        self.k_conv = nn.Conv1d(d_in, d_k, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(d_in, d_v, 1)

        self.linear_trans = nn.Conv1d(d_v, d_g, 1)
        self.bn = nn.BatchNorm1d(d_g)

    def forward(self, x):
        max_feature = torch.max(x, dim=2, keepdim=True)[0].to(x.device)  # (B, d_in, 1)
        # global_feature = torch.ones(x.size(0), x.size(1), 1).to(x.device)
        x_q = self.q_conv(max_feature).permute(0, 2, 1)     # (B, 1, d_k)
        x_k = self.k_conv(x)                                # (B, d_k, N)
        x_v = self.v_conv(x)                                # (B, d_v, N)

        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.d_k))  # (B, 1, N)
        attention = F.softmax(energy, dim=-1)                 # (B, 1, N)

        x_s = torch.bmm(attention, x_v.permute(0, 2, 1)).permute(0, 2 ,1)  # (B, d_v, 1)
        # resudual connection, it need d_in == d_v
        x_s = x_s + max_feature

        x_s = F.leaky_relu(self.bn(self.linear_trans(x_s)), negative_slope=0.2)         # (B, d_g, 1)

        x_s = x_s.view(x.size(0), -1)

        return x_s


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


# class Segmentation(nn.Module):
#     def __init__(self, part_num):
#         super().__init__()

#         self.part_num = part_num

#         self.label_conv = nn.Sequential(
#             nn.Conv1d(16, 64, kernel_size=1, bias=False),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(negative_slope=0.2)
#         )

#         self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
#         self.convs2 = nn.Conv1d(512, 256, 1)
#         self.convs3 = nn.Conv1d(256, self.part_num, 1)

#         self.bns1 = nn.BatchNorm1d(512)
#         self.bns2 = nn.BatchNorm1d(256)

#         self.dp1 = nn.Dropout(0.5)
    
#     def forward(self, x, x_max, x_mean, cls_label):
#         batch_size, _, N = x.size()

#         x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
#         x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

#         cls_label_one_hot = cls_label.view(batch_size, 16, 1)
#         cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

#         x = torch.cat([x, x_max_feature, x_mean_feature, cls_label_feature], dim=1)  # 1024 * 3 + 64

#         x = F.relu(self.bns1(self.convs1(x)))
#         x = self.dp1(x)
#         x = F.relu(self.bns2(self.convs2(x)))
#         x = self.convs3(x)

#         return x


# class NormalEstimation(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
#         self.convs2 = nn.Conv1d(512, 256, 1)
#         self.convs3 = nn.Conv1d(256, 3, 1)

#         self.bns1 = nn.BatchNorm1d(512)
#         self.bns2 = nn.BatchNorm1d(256)

#         self.dp1 = nn.Dropout(0.5)
    
#     def forward(self, x, x_max, x_mean):
#         N = x.size(2)

#         x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
#         x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)
        
#         x = torch.cat([x_max_feature, x_mean_feature, x], dim=1)

#         x = F.relu(self.bns1(self.convs1(x)))
#         x = self.dp1(x)
#         x = F.relu(self.bns2(self.convs2(x)))
#         x = self.convs3(x)

#         return x


if __name__ == '__main__':
    """
    Please be careful to excute the testing code, because
    it may cause the GPU out of memory.
    """
    
    pc = torch.rand(32, 3, 1024).to('cuda')

    # # testing for Embedding
    # embedding = Embedding().to('cuda')
    # out = embedding(pc)
    # print("Embedding output size:", out.size())

    # # testing for SA
    # sa = SA(channels=out.size(1)).to('cuda')
    # out = sa(out)
    # print("SA output size:", out.size())

    # # testing for SG
    # coords = torch.rand(32, 1024, 3).to('cuda')
    # features = torch.rand(32, 64, 1024).to('cuda')
    # sg = SG(512, 128, 128).to('cuda')
    # new_coords, out = sg(features, coords)
    # print("SG output size:", new_coords.size(), out.size())

    # testing for NeighborEmbedding
    ne = NeighborEmbedding().to('cuda')
    out = ne(pc)
    print("NeighborEmbedding output size:", out.size())

    # # testing for OA
    # oa = OA(256).to('cuda')
    # out = oa(out)
    # print("OA output size:", out.size())
