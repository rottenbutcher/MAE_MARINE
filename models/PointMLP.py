import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS

# PointMLP의 핵심 블록 정의
class ResMlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, int(in_dim * mlp_ratio), 1)
        self.conv2 = nn.Conv2d(int(in_dim * mlp_ratio), in_dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = shortcut + x
        return x

# 메인 모델 아키텍처
@MODELS.register_module()
class PointMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_dim = int(config.cls_dim)
        
        # 특징 추출을 위한 초기 컨볼루션 레이어
        self.conv_pre = nn.Conv1d(3, 32, 1)
        self.bn_pre = nn.BatchNorm1d(32)

        # Residual MLP 블록들
        self.mlp_blocks = nn.Sequential(
            ResMlpBlock(32),
            ResMlpBlock(32)
        )
        
        # 중간 특징 확장 레이어
        self.conv_mid = nn.Conv1d(32, 128, 1)
        self.bn_mid = nn.BatchNorm1d(128)

        # 더 깊은 Residual MLP 블록들
        self.mlp_blocks2 = nn.Sequential(
            ResMlpBlock(128),
            ResMlpBlock(128)
        )
        
        # 최종 특징 추출
        self.conv_post = nn.Conv1d(128, 1024, 1)
        self.bn_post = nn.BatchNorm1d(1024)

        # 분류를 위한 MLP 헤드
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )
        
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, pts):
        if self.training and pts.shape[0] <= 1:
            return torch.empty(0)
        # pts shape: [B, N, 3], 모델 입력 형식에 맞게 [B, 3, N]으로 변경
        pts = pts.transpose(2, 1)
        
        # 특징 추출 시작
        out = F.gelu(self.bn_pre(self.conv_pre(pts)))
        
        # Residual MLP 블록 통과 (2D Conv이므로 차원 확장 필요)
        out = out.unsqueeze(-1) # [B, C, N, 1]
        out = self.mlp_blocks(out)
        out = out.squeeze(-1) # [B, C, N]
        
        # 중간 특징 확장
        out = F.gelu(self.bn_mid(self.conv_mid(out)))
        
        # 두 번째 MLP 블록 통과
        out = out.unsqueeze(-1)
        out = self.mlp_blocks2(out)
        out = out.squeeze(-1)
        
        # 최종 특징 추출 및 Global Max Pooling
        out = F.gelu(self.bn_post(self.conv_post(out)))
        out = F.adaptive_max_pool1d(out, 1).squeeze(-1) # [B, C]

        # --- 최종 수정 코드 시작 ---
        # BatchNorm 오류를 막기 위한 최종 방어 코드
        # 현재 GPU에 할당된 배치의 크기가 1 이하이고, 모델이 학습 모드일 경우,
        # BatchNorm 오류를 피하기 위해 classifier를 통과하지 않고 바로 0을 반환합니다.
        if self.training and out.shape[0] <= 1:
            # 출력 형태를 맞춰주기 위해 0으로 채워진 텐서를 반환합니다.
            return torch.zeros(out.shape[0], self.cls_dim, device=out.device)
        # --- 최종 수정 코드 끝 ---

        # 배치 크기가 2 이상이면 정상적으로 classifier를 통과합니다.
        return self.classifier(out)