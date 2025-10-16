import torch
import torch.nn as nn
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2
from .Point_MAE import Group, Encoder as PointMAE_Encoder # Point-MAE의 모듈 재사용

@MODELS.register_module()
class DiscreteVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Encoder: 포인트 클라우드 패치를 feature vector로 변환
        self.encoder = PointMAE_Encoder(encoder_channel=config.encoder_dims)

        # Codebook: feature vector를 가장 가까운 이산적인 토큰으로 양자화(quantize)
        self.num_tokens = config.num_tokens
        self.codebook = nn.Embedding(self.num_tokens, config.encoder_dims)
        
        # Decoder: 토큰으로부터 다시 포인트 클라우드 패치를 복원
        self.decoder = nn.Sequential(
            nn.Conv1d(config.encoder_dims, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 3 * config.group_size, 1)
        )
        
        self.group_size = config.group_size
        self.group_divider = Group(num_group=config.num_group, group_size=self.group_size)
        
        self.rec_loss = ChamferDistanceL2()
        
    def forward(self, pts, **kwargs):
        # 1. 포인트 클라우드를 패치로 나눔
        # ================================================================= #
        #              반환값을 3개 받도록 수정 (center, idx 무시)            #
        # ================================================================= #
        neighborhood, _, _ = self.group_divider(pts)
        
        B, G, K, _ = neighborhood.shape # Batch, Group, K-points
        
        # 2. Encoder로 각 패치의 feature 추출
        patch_features = self.encoder(neighborhood) # B, G, C
        
        # 3. Codebook에서 가장 가까운 토큰 찾기 (양자화)
        dist = torch.cdist(patch_features, self.codebook.weight, p=2) # B, G, num_tokens
        token_ids = torch.argmin(dist, dim=-1) # B, G
        
        # 4. 선택된 토큰에 해당하는 codebook vector 가져오기
        quantized_features = self.codebook(token_ids) # B, G, C
        
        # 5. Decoder로 패치 복원
        recons_patches = self.decoder(quantized_features.transpose(1, 2)) # B, 3*K, G
        recons_patches = recons_patches.transpose(1, 2).reshape(B * G, K, 3) # B*G, K, 3
        
        # 6. 복원 손실(Reconstruction Loss) 계산
        gt_patches = neighborhood.reshape(B * G, K, 3)
        loss = self.rec_loss(recons_patches, gt_patches)
        
        return loss