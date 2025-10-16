import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2
from utils.logger import *
from .Point_MAE import Group
from .modules import Token_Embed, Encoder_Block, Decoder_Block
from segmentation.pointnet_util import PointNetFeaturePropagation
from knn_cuda import KNN

class H_Encoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.encoder_depths = config.encoder_depths
        self.encoder_dims = config.encoder_dims
        self.local_radius = config.get('local_radius', [0.0] * len(self.encoder_dims))

        self.token_embed = nn.ModuleList([
            Token_Embed(in_c=3 if i == 0 else self.encoder_dims[i - 1], out_c=self.encoder_dims[i])
            for i in range(len(self.encoder_dims))
        ])
        self.encoder_pos_embeds = nn.ModuleList([
            nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
            for dim in self.encoder_dims
        ])

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.encoder_depths))]
        depth_count = 0
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                embed_dim=self.encoder_dims[i], depth=self.encoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in self.encoder_dims])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, neighborhoods, centers, idxs, initial_mask):
        bool_masked_pos = [initial_mask]
        for i in range(len(neighborhoods) - 1, 0, -1):
            b, g, k, _ = neighborhoods[i].shape
            idx = idxs[i].reshape(b * g, -1)
            visible_mask = ~bool_masked_pos[-1].reshape(-1)
            visible_indices = torch.where(visible_mask)[0]
            sub_indices = idx[visible_indices].reshape(-1).unique()
            masked_pos = torch.ones(b * centers[i - 1].shape[1], device=initial_mask.device, dtype=torch.bool)
            masked_pos.scatter_(0, sub_indices.long(), 0)
            bool_masked_pos.append(masked_pos.reshape(b, centers[i - 1].shape[1]))

        bool_masked_pos.reverse()
        x_vis_list, x_vis_masks_list = [], []
        full_tokens = None

        for i in range(len(centers)):
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            else:
                b_size, n_groups_prev, _ = full_tokens.shape
                n_groups_curr, n_neighbors_curr = idxs[i].shape[1], idxs[i].shape[2]
                
                flat_tokens_prev = full_tokens.reshape(b_size * n_groups_prev, -1)
                flat_indices_base = torch.arange(b_size, device=idxs[i].device).view(-1, 1, 1) * n_groups_prev
                flat_indices = (idxs[i] + flat_indices_base).reshape(b_size * n_groups_curr, -1)
                
                neighborhood_features = flat_tokens_prev[flat_indices]
                neighborhood_features = neighborhood_features.reshape(b_size, n_groups_curr, n_neighbors_curr, -1)
                
                group_input_tokens = self.token_embed[i](neighborhood_features)

            full_tokens = group_input_tokens
            bool_vis_pos = ~bool_masked_pos[i]

            x_vis_batch = [full_tokens[b_idx][bool_vis_pos[b_idx]] for b_idx in range(full_tokens.size(0))]
            center_vis_batch = [centers[i][b_idx][bool_vis_pos[b_idx]] for b_idx in range(centers[i].size(0))]
            
            x_vis = nn.utils.rnn.pad_sequence(x_vis_batch, batch_first=True, padding_value=0.0)
            center_vis = nn.utils.rnn.pad_sequence(center_vis_batch, batch_first=True, padding_value=0.0)
            
            attn_mask = torch.zeros(x_vis.shape[:2], dtype=torch.bool, device=x_vis.device)
            for b_idx, length in enumerate([len(v) for v in x_vis_batch]):
                attn_mask[b_idx, length:] = True
            
            pos = self.encoder_pos_embeds[i](center_vis)
            x_vis = self.encoder_blocks[i](x_vis, pos, attn_mask=attn_mask.unsqueeze(1) * attn_mask.unsqueeze(2))

            temp_x_vis = x_vis.clone()
            full_tokens[bool_vis_pos] = temp_x_vis[~attn_mask]
            
            x_vis_list.append(x_vis)
            x_vis_masks_list.append(~attn_mask)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        return x_vis_list, x_vis_masks_list, bool_masked_pos


@MODELS.register_module()
class Point_M2AE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE] build ...', logger='Point_M2AE')
        self.config = config
        
        self.group_sizes = self.config.group_sizes
        self.num_groups = self.config.num_groups
        self.group_dividers = nn.ModuleList([
            Group(num_group=self.num_groups[i], group_size=self.group_sizes[i])
            for i in range(len(self.group_sizes))
        ])

        self.h_encoder = H_Encoder(self.config)

        self.decoder_depths = self.config.decoder_depths
        self.decoder_dims = self.config.decoder_dims
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder_pos_embeds = nn.ModuleList([
            nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
            for dim in self.decoder_dims
        ])
        
        dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, sum(self.decoder_depths))]
        depth_count = 0
        self.h_decoder = nn.ModuleList()
        for i in range(len(self.decoder_dims)):
            self.h_decoder.append(Decoder_Block(
                embed_dim=self.decoder_dims[i], depth=self.decoder_depths[i],
                drop_path_rate=dpr[depth_count : depth_count + self.decoder_depths[i]],
                num_heads=self.config.num_heads,
            ))
            depth_count += self.decoder_depths[i]

        self.token_prop = nn.ModuleList([
            PointNetFeaturePropagation(
                in_channel=self.decoder_dims[i-1] + self.decoder_dims[i], mlp=[self.decoder_dims[i]]
            ) for i in range(1, len(self.decoder_dims))
        ])

        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        self.rec_loss = ChamferDistanceL2().cuda()

    def forward(self, partial_view, ground_truth, **kwargs):
        neighborhoods, centers, idxs = [], [], []
        pts = ground_truth
        for i in range(len(self.group_dividers)):
            neighborhood, center, idx = self.group_dividers[i](pts)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)
            pts = center

        B, G, _ = centers[-1].shape
        knn = KNN(k=1, transpose_mode=True)
        _, idx = knn(centers[-1], partial_view)
        initial_mask = torch.ones((B, G), dtype=torch.bool, device=ground_truth.device)
        for b in range(B):
            initial_mask[b, torch.unique(idx[b])] = False
            
        x_vis_list, x_vis_masks_list, masks = self.h_encoder(neighborhoods, centers, idxs, initial_mask)
        
        centers.reverse()
        neighborhoods.reverse()
        x_vis_list.reverse()
        masks.reverse()
        x_vis_masks_list.reverse()
        
        x_full = None
        for i in range(len(self.decoder_dims)):
            center_level = centers[i]
            bool_mask_pos = masks[i]
            
            if i == 0:
                x_vis, x_vis_mask = x_vis_list[i], x_vis_masks_list[i]
                B, _, C_vis = x_vis.shape
                G_level = center_level.shape[1]
                C_dec = self.decoder_dims[i]
                
                x_full = self.mask_token.expand(B, G_level, C_dec).clone()
                
                # 보이는 위치에 인코더 출력을 정확히 배치
                full_indices = torch.where(~bool_mask_pos)
                vis_indices = torch.where(x_vis_mask)
                x_full[full_indices] = x_vis[vis_indices]

                pos_full = self.decoder_pos_embeds[i](center_level)
            else:
                pos_full = self.decoder_pos_embeds[i](center_level)
                x_full = self.token_prop[i - 1](
                    center_level.transpose(1, 2), centers[i - 1].transpose(1, 2), 
                    pos_full.transpose(1, 2), x_full.transpose(1, 2)
                ).transpose(1, 2)

            x_full = self.h_decoder[i](x_full, pos_full)

        x_full = self.decoder_norm(x_full)
        
        # ================================================================= #
        #              가장 세밀한 레벨의 마스크(masks[-1]) 사용                #
        # ================================================================= #
        final_mask = masks[-1]

        num_rec_patches = final_mask.sum()
        if num_rec_patches == 0:
            return torch.tensor(0.0, device=ground_truth.device, requires_grad=True)

        rec_patch_tokens = x_full[final_mask]
        rec_points = self.rec_head(rec_patch_tokens.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        rec_points = rec_points.reshape(num_rec_patches, self.group_sizes[0], 3)
        gt_points = neighborhoods[-1][final_mask]
        
        loss = self.rec_loss(rec_points, gt_points)
        return loss