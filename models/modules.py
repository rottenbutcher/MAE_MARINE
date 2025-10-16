import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

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
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            # attn_mask is (B, N, N), need to expand for multi-head
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn = attn.masked_fill(attn_mask, -1e9) # 마스크가 True인 위치에 아주 작은 값을 더해 softmax 후 0이 되도록 함
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, pos, attn_mask=None):
        x = x + pos
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Token_Embed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        if in_c == 3:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1)
            )
            self.second_conv = nn.Sequential(
                nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, out_c, 1)
            )
        else:
            self.first_conv = nn.Sequential(
                nn.Conv1d(in_c, in_c, 1), nn.BatchNorm1d(in_c), nn.ReLU(inplace=True), nn.Conv1d(in_c, out_c, 1)
            )
    
    def forward(self, point_groups):
        bs, g, k, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, k, c)
        if self.in_c == 3:
            feature = self.first_conv(point_groups.transpose(2, 1))
            feature_global = torch.max(feature, dim=2, keepdim=True)[0]
            feature = torch.cat([feature_global.expand(-1, -1, k), feature], dim=1)
            feature = self.second_conv(feature)
        else:
            feature = self.first_conv(point_groups.transpose(2, 1))
        
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, -1)

class Encoder_Block(nn.Module):
    def __init__(self, embed_dim, depth, drop_path_rate, num_heads):
        super().__init__()
        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=drop_path_rate[i]
        ) for i in range(depth)])
    
    def forward(self, x, pos, attn_mask=None):
        for block in self.blocks:
            x = block(x, pos, attn_mask)
        return x

class Decoder_Block(nn.Module):
    def __init__(self, embed_dim, depth, drop_path_rate, num_heads):
        super().__init__()
        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=drop_path_rate[i]
        ) for i in range(depth)])
    
    def forward(self, x, pos, attn_mask=None):
        for block in self.blocks:
            x = block(x, pos, attn_mask)
        return x