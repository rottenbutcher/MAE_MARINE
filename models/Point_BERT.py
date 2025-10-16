import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
# Point-BERT는 dVAE의 Group과 Encoder를 사용하므로, 별도의 dvae.py 파일이 필요합니다.
# 이 코드는 Point-MAE 프로젝트의 구조를 따르므로, 관련 모듈을 임시로 가져옵니다.
from .Point_MAE import Group, Encoder as PointMAE_Encoder

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN

# --- Transformer 기본 블록 (Mlp, Attention, Block, TransformerEncoder) ---
# 제공해주신 코드와 동일하게 유지됩니다. 아래에 전체 코드를 포함시켰습니다.
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

# --- Point-BERT의 MaskTransformer ---
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # MaskTransformer가 받는 config는 이미 model 섹션임
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims =  config.encoder_dims
        
        self.encoder = PointMAE_Encoder(encoder_channel = self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        
        self.num_tokens = config.num_tokens
        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)

        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighborhood, center, custom_mask=None, **kwargs):
        bool_masked_pos = custom_mask
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)

        batch_size, seq_len, C = group_input_tokens.size()
        
        x_full = torch.cat([self.cls_token.expand(batch_size, -1, -1), group_input_tokens], dim=1)
        
        pos_full = self.pos_embed(center)
        pos_full = torch.cat([self.cls_pos.expand(batch_size, -1, -1), pos_full], dim=1)

        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x_full[:, 1:] = x_full[:, 1:] * (1 - w) + mask_token * w

        x = self.blocks(x_full, pos_full)
        x = self.norm(x)

        logits = self.lm_head(x[:, 1:])
        masked_logits = logits[bool_masked_pos]
        return masked_logits

# --- Point-BERT 메인 모델 ---
@MODELS.register_module()
class Point_BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Point_BERT 생성자에 들어오는 config는 yaml의 'model' 섹션임
        print_log(f'[Point_BERT] build ...', logger ='Point_BERT')
        self.config = config
        
        # 'config.model' 이 아니라 'config'에서 직접 값을 가져오도록 수정
        self.num_tokens = config.get('num_tokens', 8192)
        
        # 'config.model' 대신 'config'를 전달하도록 수정
        self.transformer_q = MaskTransformer(config)
        
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, partial_view, ground_truth, **kwargs):
        neighborhood, center = self.group_divider(ground_truth)
        
        B, G, _, _ = neighborhood.shape
        dvae_label = torch.arange(G, device=ground_truth.device).unsqueeze(0).expand(B, -1)

        knn = KNN(k=1, transpose_mode=True)
        # partial_view의 각 점에 대해 center 중에서 가장 가까운 점의 인덱스를 찾음
        _, idx = knn(center, partial_view)
        
        bool_masked_pos = torch.ones((B, G), dtype=torch.bool, device=ground_truth.device)
        for b in range(B):
            unique_vis_idx = torch.unique(idx[b])
            bool_masked_pos[b, unique_vis_idx] = False

        masked_logits = self.transformer_q(neighborhood, center, custom_mask=bool_masked_pos)

        masked_dvae_labels = dvae_label[bool_masked_pos]
        loss = self.loss_ce(masked_logits, masked_dvae_labels.long())
        
        return loss