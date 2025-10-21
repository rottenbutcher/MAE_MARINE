import numpy as np
import torch
import random
import torch.nn.functional as F
import math

class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc

class RandomHorizontalFlip(object):


  def __init__(self, upright_axis = 'z', is_temporal=False):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])


  def __call__(self, coords):
    bsize = coords.size()[0]
    for i in range(bsize):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(coords[i, :, curr_ax])
                    coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
    return coords
  


class PointcloudViewpointMasking(object):
    def __init__(self, viewpoint_mask_ratio=0.5, random_mask_ratio=0.3):
        self.viewpoint_mask_ratio = viewpoint_mask_ratio
        self.random_mask_ratio = random_mask_ratio

    def __call__(self, pc):
        """
        pc: Tensor of shape (B, N, 3)
        """
        B, N, _ = pc.shape
        
        # 최종 마스크를 저장할 텐서
        final_mask = torch.zeros(B, N, dtype=torch.bool, device=pc.device)

        # 배치 내의 각 포인트 클라우드에 대해 개별적으로 처리
        for i in range(B):
            points = pc[i] # (N, 3)

            # --- 1. 주방향 계산 (PCA) ---
            # 포인트 클라우드의 중심을 원점으로 이동
            centered_points = points - points.mean(dim=0, keepdim=True)
            # 공분산 행렬 계산
            cov = torch.matmul(centered_points.T, centered_points) / (N - 1)
            # 고유값(eigenvalues), 고유벡터(eigenvectors) 계산
            _, eigenvectors = torch.linalg.eigh(cov)
            # 가장 큰 고유값에 해당하는 고유벡터가 주방향
            principal_axis = eigenvectors[:, -1] # shape (3,)

            # --- 2. 유효한 Viewpoint 생성 ---
            # 높이 각도 (elevation): -20 ~ +20도
            elevation_angle = torch.deg2rad(torch.FloatTensor(1).uniform_(-20, 20))
            
            # 수평 각도 (azimuth) 샘플링
            while True:
                azimuth_angle_deg = torch.FloatTensor(1).uniform_(0, 360)
                if not (azimuth_angle_deg <= 20 or (azimuth_angle_deg >= 160 and azimuth_angle_deg <= 200) or azimuth_angle_deg >= 340):
                    break
            azimuth_angle = torch.deg2rad(azimuth_angle_deg)

            # 구면 좌표계를 사용하여 Viewpoint 벡터 생성
            x = torch.cos(azimuth_angle) * torch.cos(elevation_angle)
            y = torch.sin(azimuth_angle) * torch.cos(elevation_angle)
            z = torch.sin(elevation_angle)
            
            # 주방향을 z축으로 회전시키는 행렬 계산
            # 이 과정은 Viewpoint를 객체 좌표계에 맞게 정렬합니다.
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=pc.device)
            v = torch.cross(z_axis, principal_axis)
            c = torch.dot(z_axis, principal_axis)
            s = torch.linalg.norm(v)
            kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=pc.device)
            rotation_matrix = torch.eye(3, device=pc.device) + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2))

            # Viewpoint를 회전시켜 객체 좌표계에 맞춤
            viewpoint = torch.matmul(rotation_matrix, torch.tensor([x, y, z], device=pc.device)).unsqueeze(0) # (1, 3)

            # --- 3. 마스킹 수행 ---
            dot_product = torch.sum(centered_points * viewpoint, dim=-1) # (N,)
            
            num_viewpoint_masked = int(N * self.viewpoint_mask_ratio)
            sorted_indices = torch.argsort(dot_product)
            viewpoint_masked_indices = sorted_indices[:num_viewpoint_masked]
            
            viewpoint_mask = torch.zeros(N, dtype=torch.bool, device=pc.device)
            viewpoint_mask[viewpoint_masked_indices] = True
            
            visible_indices = sorted_indices[num_viewpoint_masked:]
            num_remaining = N - num_viewpoint_masked
            num_random_masked = int(num_remaining * self.random_mask_ratio)
            
            shuffled_visible_indices = visible_indices[torch.randperm(num_remaining)]
            random_masked_indices = shuffled_visible_indices[:num_random_masked]
            
            random_mask = torch.zeros(N, dtype=torch.bool, device=pc.device)
            if num_random_masked > 0:
                random_mask[random_masked_indices] = True

            final_mask[i] = viewpoint_mask | random_mask

        masked_pc = pc.clone()
        masked_pc[final_mask] = 0.0

        return masked_pc