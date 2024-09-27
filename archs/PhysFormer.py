"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math

# from archs.transformer_layer import Transformer_ST_TDC_gra_sharp
from archs.transformer_layer import Transformer_ST_MTV_Gated_Dconv
from archs.transformer_layer import Cross_View_Attention
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import numpy as np


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

def get_sinusoid_encoding_table(n_position, d_hid):  
    # n_position = 5120 = (160/8)*(128/8)*(128/8)
    # d_hid = 384
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    # [5120, 384]

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 

class Stem(nn.Module):
    def __init__(self,patch_size):
        super().__init__()

        self.dim = 96
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, self.dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(self.dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)), 
        )  # [B, 16, 160, 256, 256]
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(self.dim//4, self.dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(self.dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # [B, 32, 160, 128, 128]
        self.Stem2 = nn.Sequential(
            nn.Conv3d(self.dim//2, self.dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(self.dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # [B, 64, 160, 64, 64]
        self.patch_embedding = nn.Conv3d(self.dim, self.dim, kernel_size=(patch_size, patch_size, patch_size), stride=(patch_size, patch_size, patch_size))

    def forward(self, x):
        b, c, t, fh, fw = x.shape
        # print("stem:")
        x = self.Stem0(x)  # [2, 24, 300, 64, 64]
        # print("x:" + str(x.shape))
        x = self.Stem1(x)  # [2, 48, 300, 32, 32]
        # print("x:" + str(x.shape))
        x = self.Stem2(x)  # [2, 96, 300, 16, 16]
        # print("stem")
        # print(x.shape)
        # print("x:" + str(x.shape))

        # print("patch_embedding:")
        x = self.patch_embedding(x)  # [2, 96, 75, 4, 4]
        # print(x.shape)
        # print("x:" + str(x.shape))
        x = x.flatten(2).transpose(1, 2)  # [2, 1200, 96]
        # print(x.shape)
        # print(x.shape)
        # print("x:" + str(x.shape))
        return x

# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class Pretrain_ViT_ST_ST_Compact3_TDC_gra_sharp_Encoder(nn.Module):

    def __init__(
        self, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout_rate: float = 0.2,
        #positional_embedding: str = '1d',
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
    ):
        super().__init__()

        
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim  # dim = 64    
        self.num_class = 0           

        # Image and patch sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40

        # Patch embedding -- (ft, fh, fw) = (4, 16, 16)  
        # [B, 64, 160, 64, 64]
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        # [B, 64, 40, 4, 4]
        
        # Transformer
        # num_layers=num_layers//3 : 用于得到Score123
        # [B, 4*4*40, 64]
        self.transformer1 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)  # [B, 4*4*40, 64]
        # Transformer
        self.transformer2 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)  # [B, 4*4*40, 64]
        # Transformer
        self.transformer3 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate, theta=theta)  # [B, 4*4*40, 64]
        
        # [B, 16, 160, 512, 512]
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)), 
        )  # [B, 16, 160, 256, 256]
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # [B, 32, 160, 128, 128]
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # [B, 64, 160, 64, 64]
        # Stem中只有nn.MaxPool3d模块改变Tensor shape
        
        #self.normLast = nn.LayerNorm(dim, eps=1e-6)
        
        # [B, 64, 40, 4, 4]
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1)),
            nn.Conv2d(dim, dim, [3, 1], stride=1, padding=(1,0)),   
            nn.BatchNorm2d(dim),
            nn.ELU(),
        )  # [B, 64, 80, 4, 4]
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1)),
            nn.Conv2d(dim, dim//2, [3, 1], stride=1, padding=(1,0)),   
            nn.BatchNorm2d(dim//2),
            nn.ELU(),
        )  # [B, 32, 160, 4, 4]
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        
        
        # Initialize weights
        self.init_weights()

        self.pos_embed_num_patches = (frame // 4) * (image_size // 32) * (image_size // 32)
        # self.pos_embed_num_patches = 75 * (image_size // 32) * (image_size // 32) #####
        self.pos_embed = get_sinusoid_encoding_table(self.pos_embed_num_patches, self.dim)
        self.norm =  nn.LayerNorm(self.dim)
        # self.head = nn.Linear(self.dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Identity()


        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, mask):
        # print("x:" + str(x.shape))
        # pos_test = self.pos_embed.type_as(x).to(x.device).clone().detach()  # [1, 5120, 384]
        # # print("pos_test:" + str(pos_test.shape))
        # x = x + pos_test  # [2, 5120, 384]
        # print("x:" + str(x.shape))

        B, _, C = x.shape
        # print("mask:" + str(mask.shape))
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible  # [2, 1280, 384]
        # print("x_vis:" + str(x_vis.shape))
        
        # print("x_vis")
        # print(x_vis.shape)
        # Temporal Difference Transformer
        gra_sharp = 2.0
        Trans_features, Score1 =  self.transformer1(x_vis, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features2, Score2 =  self.transformer2(Trans_features, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features3, Score3 =  self.transformer3(Trans_features2, gra_sharp)  # [B, 4*4*40, 64]
        # print(Trans_features3.shape)
        #Trans_features3 = self.normLast(Trans_features3)
        # print("Trans_features3:" + str(Trans_features3.shape))

        x_result = self.norm(Trans_features3)
        x_result = self.head(x_result)
        # print("x_result:" + str(x_result.shape))
        
        # upsampling heads
        #features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 40, 4, 4) # [B, 64, 40, 4, 4]
        # features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t//4, 4, 4) # [B, 64, 40, 4, 4]
        # print("x_result:")
        # print(x_result.shape)
        features_last = Trans_features3.transpose(1, 2).view(B, self.dim, 40, -1) #####
        # features_last = Trans_features3.transpose(1, 2).view(B, self.dim, 75, -1)
        # print(features_last.shape)
        features_last = self.upsample(features_last)		    # [B, 64, 80, 4, 4]
        features_last = self.upsample2(features_last)		    # [B, 64, 160, 4, 4]
        # print(features_last.shape)
        
        features_last = torch.mean(features_last,3)     # x [B, 32, 160, 4]  
        # print(features_last.shape)
        
        # features_last = torch.mean(features_last,3)     # x [B, 32, 160]    
        rPPG = self.ConvBlockLast(features_last)    # x [B, 1, 160]
        # print(rPPG.shape)
        

        rPPG = rPPG.squeeze(1)  # x [B, 160]
        # print(rPPG.shape)
        
        return x_result, rPPG

class Pretrain_ViT_ST_ST_Compact3_TDC_gra_sharp_Decoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=4, num_classes=96, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        # print("decoder:")
        for blk in self.blocks:
            x = blk(x)  # [2, 5120, 192]
        # print("blk(x):" + str(x.shape))

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels  # [2, 3840, 1536]
        else:
            x = self.head(self.norm(x))
        # print("head(x):" + str(x.shape))

        return x

class Pretrain_ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                patches: int = 16,
                dim: int = 768,
                ff_dim: int = 3072,
                num_heads: int = 12,
                num_layers: int = 12,
                dropout_rate: float = 0.2,
                #positional_embedding: str = '1d',
                frame: int = 160,
                theta: float = 0.2,
                image_size: Optional[int] = None,
                init_values=0.,
                encoder_embed_dim=96, 
                decoder_embed_dim=192,
                mask_ratio=0.5,
    ):
        super().__init__()
        self.encoder = Pretrain_ViT_ST_ST_Compact3_TDC_gra_sharp_Encoder(
            patches = patches,
            image_size=image_size,
            dim = dim,
            ff_dim = ff_dim,
            num_heads = num_heads,
            num_layers = num_layers,
            dropout_rate = dropout_rate,
            frame = frame,
            theta = theta,
            )
        
        self.pos_embed_dim = (frame // 4) * (image_size // 32) * (image_size // 32)

        self.decoder = Pretrain_ViT_ST_ST_Compact3_TDC_gra_sharp_Decoder(init_values=init_values)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, int(self.pos_embed_dim * mask_ratio), decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.pos_embed_dim, decoder_embed_dim)
        

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        # _, _, T, _, _ = x.shape
        x_vis, rPPG = self.encoder(x, mask) # [B, N_vis, C_e]  # [2, 1280, 384]
        # print("encoder:" + str(x_vis.shape))
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]  # [2, 1280, 192]
        B, N, C = x_vis.shape
        # print("encoder_to_decoder:" + str(x_vis.shape))
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()  # [2, 5120, 192]
        # pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)  # [2, 1280, 192]
        # print("pos_emd_vis:" + str(pos_emd_vis.shape))
        # pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)  # [2, 3840, 192]
        # print("pos_emd_mask:" + str(pos_emd_mask.shape))
        # x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]  # [2, 5120, 192]
        # print("x_full:" + str(x_full.shape))

        mask_token = self.mask_token.repeat(B,1,1)
        # print("mask_token:" + str(mask_token.shape))
        x_full = torch.cat([x_vis , mask_token], dim=1) # [B, N, C_d]  # [2, 5120, 192]
        # print("x_full:" + str(x_full.shape))

        # x = self.decoder(x_full, mask_token.shape[1])  # [2, 3840, 1536]
        x = self.decoder(x_full, 0)  # [2, 3840, 1536]
        # print("self.decoder:" +str(x.shape))

        return x, rPPG

class Finetune_Physformer_MTV_GDFF(nn.Module):

    def __init__(
        self, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        # fusion_layers: int = 6,
        dropout_rate: float = 0.2,
        #positional_embedding: str = '1d',
        frame: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
    ):
        super().__init__()

        
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim  # dim = 64 
        self.num_layers = num_layers
        # self.fusion_layers = fusion_layers  

        # 手动设置
        # self.t_hidden_min_dim = 20           

        # Image and patch sizes
        t = frame
        h, w = as_tuple(image_size)  # tube sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40


        # Patch embedding -- (ft, fh, fw) = (4, 16, 16)  
        # [B, 64, 160, 64, 64]
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(4, 4, 4), stride=(4, 4, 4))
        self.patch_embedding2 = nn.Conv3d(dim, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        # [B, 64, 40, 4, 4]

        self.transformer1 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, 
                    num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer2 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, 
                    num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer3 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, 
                    num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer4 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, 
                    num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        self.transformer5 = Transformer_ST_MTV_Gated_Dconv(num_layers=num_layers//3, dim=dim, 
                    num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate, theta=theta)

        self.cross_view_transformer1 = Cross_View_Attention(dim=dim, num_heads=num_heads, 
                    ff_dim=ff_dim, dropout=dropout_rate)
        self.cross_view_transformer2 = Cross_View_Attention(dim=dim, num_heads=num_heads, 
                    ff_dim=ff_dim, dropout=dropout_rate)

        
        # [B, 16, 160, 512, 512]
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim//4, [1, 5, 5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(dim//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)), 
        )  # [B, 16, 160, 256, 256]
        
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim//4, dim//2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # [B, 32, 160, 128, 128]
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim//2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )  # [B, 64, 160, 64, 64]
        # Stem中只有nn.MaxPool3d模块改变Tensor shape
        
        #self.normLast = nn.LayerNorm(dim, eps=1e-6)
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim),
            nn.ELU(),
        ) 
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(dim, dim//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(dim//2),
            nn.ELU(),
        )
 
        self.ConvBlockLast = nn.Conv1d(dim//2, 1, 1,stride=1, padding=0)
        
        
        # Initialize weights
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)


    def forward(self, x, gra_sharp):

        # B = 4
        b, c, t, fh, fw = x.shape  # [B, 3, 160, 128, 128]

        # Stem
        x = self.Stem0(x)  # [B, 24, 160, 64, 64]
        x = self.Stem1(x)  # [B, 48, 160, 32, 32]
        x = self.Stem2(x)  # [B, 96, 160, 16, 16]
        
        # Tube Tokens -- 2个View
        x1 = self.patch_embedding(x)  # [B, 96, 40, 4, 4]
        x2 = self.patch_embedding2(x)  # [B, 96, 80, 4, 4]

        # transpose
        x1 = x1.flatten(2).transpose(1,2)  # [B, 40*4*4, 96]
        x2 = x2.flatten(2).transpose(1,2)  # [B, 80*4*4, 96]

        x2_features_1, Score =  self.transformer4(x2, gra_sharp)  # [B, 80*4*4, 96]
        x2_features_2, Score =  self.transformer5(x2_features_1, gra_sharp)  # [B, 80*4*4, 96]  

        x1_features_1, Score1 = self.transformer1(x1, gra_sharp)  # [B, 40*4*4, 96]
        x1_crossfeatures_1, Score1 = self.cross_view_transformer1(x2_features_1, x1_features_1, gra_sharp)  # [B, 40*4*4, 96]
        x1_features_2, Score2 = self.transformer2(x1_crossfeatures_1, gra_sharp)  # [B, 40*4*4, 96]
        x1_crossfeatures_2, Score2 = self.cross_view_transformer2(x2_features_2, x1_features_2, gra_sharp)  # [B, 40*4*4, 96]
        x1_features_3, Score3 = self.transformer3(x1_crossfeatures_2, gra_sharp)  # [B, 40*4*4, 96]

        features_last = x1_features_3.transpose(1, 2).view(b, self.dim, t//4, self.image_size//32, self.image_size//32)  # [B, 96, 40, 4, 4]
        features_last = self.upsample(features_last)		    # [B, 96, 80, 4, 4]
        features_last = self.upsample2(features_last)		    # [B, 48, 160, 4, 4]
        features_last = torch.mean(features_last,3)     # x [B, 48, 160, 4]  
        features_last = torch.mean(features_last,3)     # x [B, 48, 160]   

        rPPG = self.ConvBlockLast(features_last)    # x [B, 1, 160]
        rPPG = rPPG.squeeze(1)  # x [B, 160]
        
        return rPPG, Score1, Score2, Score3

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):  # [2, 1280, 384]
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)