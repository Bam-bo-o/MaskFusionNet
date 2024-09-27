"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import torch
import math 
import numpy as np

from torch import nn
from torchvision.ops import deform_conv2d
from torch.nn import functional as F
from visualizer import get_local



def split_last(x, shape):
    # x.shape = [B, 4*4*40, 64], shape = [12, -1]
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1  # count() -- 统计字符串里某个字符或子字符串出现的次数
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        # index() -- 查找指定值的首次出现的位置;
        # np.prod() -- 所有元素的乘积
        # shape[1] = int(64 / 12) = 6
    return x.view(*x.size()[:-1], *shape) # [B, 4*4*40, 12, 6]
    #?? 多余的


def merge_last(x, n_dims):  
    # x.shape = [B, 4*4*40, 12, 6], n_dims = 2
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)  #?? [B, 4*4*40, 64(72)]

def get_drop_pattern(x):
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    probability = torch.empty(shape).uniform_(0, 1)
    # return torch.bernoulli(probability).astype('float32')
    return torch.bernoulli(probability).float()



'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, stride, stride), padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal


class MultiHeadedSelfAttention_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    # 具有深度3D卷积的多头点积注意
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            nn.Conv3d(dim, dim, 3, stride=1, padding=1, groups=1, bias=False),  
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_k = nn.Sequential(
            nn.Conv3d(dim, dim, 3, stride=1, padding=1, groups=1, bias=False),  
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),  
            #nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    # @get_local('attention_map')
    def forward(self, x, gra_sharp):    # [B, 4*4*40, 64]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        [B, P, C]=x.shape  # [B, 4*4*40, 64]
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, 64, 40, 4, 4]
        # x  = x.transpose(1, 2).view(B, C, P//4, 2, 2)      # [B, 64, 40, 4, 4]  #####
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)  # [B, 64, 40, 4, 4]
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, 64]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, 64]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, 64]
        
        # self.n_heads = 12
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # [B, 12, 4*4*40, 6]
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp  # [B, 12, 4*4*40, 4*4*40]
        # print("scores:" + str(scores.shape))
        # attention_map = scores
        # print("attention_map:" + str(attention_map.shape))
        scores = self.drop(F.softmax(scores, dim=-1))  # [B, 12, 4*4*40, 4*4*40]
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()  # [B, 4*4*40, 12, 6]
        # -merge-> (B, S, D)
        h = merge_last(h, 2)  #?? [B, 4*4*40, 64(72)]
        self.scores = scores
        return h, scores


## Gated-Dconv Feed-Forward Network (GDFN)
class GatedDconv_FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super(GatedDconv_FeedForward, self).__init__()

        # ff_dim = 3072
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim*2, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim*2),
            nn.ELU(),
        )
        
        self.dwconv = nn.Conv3d(ff_dim*2, ff_dim*2, 3, stride=1, padding=1, groups=ff_dim, bias=False)
        
        self.norm1 = nn.BatchNorm3d(ff_dim)
        self.norm2 = nn.BatchNorm3d(ff_dim)
        
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):
        [B, P, C]=x.shape  # [B, 640, 96]
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)  # [B, 96, 40, 4, 4]
        # x = x.transpose(1, 2).view(B, C, P//4, 2, 2)  # [B, 96, 40, 4, 4]
        x = self.fc1(x)  # [B, 288, 40, 4, 4]
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # [B, 144, 40, 4, 4]
        # x = F.gelu(self.norm1(x1)) * self.norm2(x2)  # [B, 144, 40, 4, 4]
        x = F.gelu(x1) * x2
        x = self.fc2(x)  # [B, 96, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 640, 96]
        return x


class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        
        # ff_dim = 3072
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )       
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )     
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):    # [B, 4*4*40, 64]
        [B, P, C]=x.shape
        #x = x.transpose(1, 2).view(B, C, 40, 4, 4)      # [B, dim, 40, 4, 4]
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, 64, 40, 4, 4]
        # x  = x.transpose(1, 2).view(B, C, P//4, 2, 2)  #####
        x = self.fc1(x)		              # x [B, 3072, 40, 4, 4]
        x = self.STConv(x)		          # x [B, 3072, 40, 4, 4]
        x = self.fc2(x)		              # x [B, 64, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, 64]
        
        return x



# Temporal Difference Transformer
class Block_ST_MTV_Gated_Dconv(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_gra_sharp(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)  # ff_dim = 3072
        # self.gdff = GatedDconv_FeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)  # [B, 4*4*40, 64]
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score


class Transformer_ST_MTV_Gated_Dconv(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_MTV_Gated_Dconv(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
            ])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


class MultiHeadedAttention_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    # 具有深度3D卷积的多头点积注意
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        
        self.proj_q = nn.Sequential(
            nn.Conv3d(dim, dim, 3, stride=1, padding=1, groups=1, bias=False),  
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_k = nn.Sequential(
            nn.Conv3d(dim, dim, 3, stride=1, padding=1, groups=1, bias=False),  
            nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),  
            #nn.BatchNorm3d(dim),
            #nn.ELU(),
        )
        
        #self.proj_q = nn.Linear(dim, dim)
        #self.proj_k = nn.Linear(dim, dim)
        #self.proj_v = nn.Linear(dim, dim)
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, y, gra_sharp):    # [B, 4*4*40, 64]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        [B_x, P_x, C_x]=x.shape  # [B, 64, 96]
        [B_y, P_y, C_y]=y.shape  # [B, 32, 96]
        x = x.transpose(1, 2).view(B_x, C_x, P_x//16, 4, 4)  # [B, 96, 4, 4, 4]
        y = y.transpose(1, 2).view(B_y, C_y, P_y//16, 4, 4)  # [B, 96, 2, 4, 4]
        # x = x.transpose(1, 2).view(B_x, C_x, P_x//4, 2, 2)  # [B, 64, 40, 4, 4]
        # y = y.transpose(1, 2).view(B_y, C_y, P_y//4, 2, 2)  # [B, 64, 20, 4, 4]
        
        q = self.proj_q(y)  # [B, 96, 2, 4, 4]
        k, v = self.proj_k(x), self.proj_v(x)  # [B, 64, 4, 4, 4]
        q = q.flatten(2).transpose(1, 2)  # [B, 32, 96]
        k = k.flatten(2).transpose(1, 2)  # [B, 64, 96]
        v = v.flatten(2).transpose(1, 2)  # [B, 64, 96]
        
        # self.n_heads = 4
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # q:[B, 4, 32, 24]   k,v:[B, 4, 64, 24]
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp  # [B, 4, 32, 64]
        scores = self.drop(F.softmax(scores, dim=-1))  # [B, 4, 32, 64]
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()  # [B, 32, 4, 24]
        # -merge-> (B, S, D)
        h = merge_last(h, 2)  # [B, 32, 96]
        self.scores = scores
        return h, scores




# MTV_Model1 Cross-view attention
class Cross_View_Attention(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn1 = MultiHeadedAttention_gra_sharp(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.proj1 = nn.Linear(dim, dim)

        self.attn2 = MultiHeadedSelfAttention_gra_sharp(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.proj2 = nn.Linear(dim, dim)

        self.norm4 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        # self.mlp = MlpBlock(dim, ff_dim, dropout)  # ff_dim = 3072
        # self.gdff = GatedDconv_FeedForward(dim, ff_dim)
        self.norm4 = nn.LayerNorm(dim, eps=1e-6)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, y, gra_sharp):
        device = x.device
        Atten1, Score = self.attn1(self.norm1(x), self.norm2(y), gra_sharp)
        Atten1 = self.drop(self.proj1(Atten1))
        drop_pattern = get_drop_pattern(Atten1).to(device)
        Atten1 = Atten1 * (1.0 - drop_pattern) + y

        Atten2, Score = self.attn2(self.norm3(Atten1), gra_sharp)  # [B, 4*4*40, 64]
        h1 = self.drop(self.proj2(Atten2))
        drop_pattern = get_drop_pattern(h1).to(device)
        h1 = h1 * (1.0 - drop_pattern) + Atten1

        h2 = self.drop(self.pwff(self.norm4(h1)))
        drop_pattern = get_drop_pattern(h2).to(device)
        h2 = h2 * (1.0 - drop_pattern) + h1

        return h2, Score


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1):
        super(DeformConv2d, self).__init__()
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding) #原卷积
 
        self.conv_offset = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        ## 2*kernel_size*kernel_size 即偏移，其参数原本为2*group*kernel_size*kernel_size，group最大是inc，表示每个通道的偏移，这里规定每个通道的偏移相同（图像级别）
        self.conv_offset.weight.data.zero_()
 
        self.conv_mask = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_mask.weight.data.fill_(0.5) #初始化为0.5
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) #保证在0到1之间
        out = deform_conv2d(input=x, offset=offset, 
                            weight=self.conv.weight, 
                            mask=mask, stride=self.stride, padding=self.padding)
        return out