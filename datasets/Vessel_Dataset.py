import os
import torch
import numpy as np
import torch.utils.data as data
import open3d as o3d
from .io import IO
from .build import DATASETS
from utils.logger import print_log

# Open3D의 상세 로그 메시지를 비활성화하여 콘솔을 깔끔하게 유지합니다.
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

@DATASETS.register_module()
class Vessel(data.Dataset):
    def __init__(self, config):
        """
        Vessel 데이터셋을 위한 커스텀 데이터 로더입니다.
        
        - 사전학습(Pre-training) 시: HPR을 적용하여 (partial_view, ground_truth) 쌍을 생성합니다.
        - 미세조정(Fine-tuning) 시: 전체 포인트 클라우드와 레이블을 반환합니다.
        """
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS # 예: 8192 (Ground Truth 포인트 수)
        
        # HPR 관련 설정값들을 config에서 가져옵니다.
        self.hpr_points = config.get('HPR_POINTS', 2458) # Partial view 포인트 수
        self.vertical_angle_limit = config.get('HPR_VERTICAL_ANGLE', 20)
        self.exclusion_angle = config.get('HPR_EXCLUSION_ANGLE', 20)
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        
        print_log(f'[DATASET] Loading data from {self.data_list_file}', logger='Vessel')
        print_log(f'[DATASET] Ground truth points: {self.npoints}', logger='Vessel')

        try:
            with open(self.data_list_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print_log(f'ERROR: {self.data_list_file} not found!', logger='Vessel')
            raise

        self.file_list = []
        self.labels = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                self.file_list.append(parts[0])
                if len(parts) > 1:
                    self.labels.append(int(parts[1]))

        if len(self.labels) == 0:
            print_log(f'[DATASET] Pre-training mode enabled. Generating HPR views with {self.hpr_points} points.', logger='Vessel')
            self.is_finetuning = False
        elif len(self.labels) == len(self.file_list):
            print_log(f'[DATASET] Fine-tuning mode enabled. {len(self.labels)} labels loaded.', logger='Vessel')
            self.is_finetuning = True
        else:
            raise ValueError("ERROR: Mismatch between number of files and labels.")

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded for {self.subset}.', logger='Vessel')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC - 포인트 클라우드를 단위 구 내에 정규화합니다. """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def random_sample(self, pc, num):
        """ 포인트 클라우드를 주어진 개수(num)만큼 무작위 샘플링합니다. """
        if pc.shape[0] < num:
            indices = np.random.choice(pc.shape[0], num, replace=True)
        else:
            indices = np.random.choice(pc.shape[0], num, replace=False)
        return pc[indices]
    
    def _generate_hpr_view(self, gt_pc):
        """
        주어진 Ground Truth 포인트 클라우드로부터 HPR 뷰를 생성합니다.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_pc)
        center = pcd.get_center()

        # 1. 지향성 경계 상자(OBB)를 통해 주방향 추출
        try:
            oriented_bbox = pcd.get_oriented_bounding_box()
            main_axis_3d = oriented_bbox.R[:, 0]  # 가장 긴 축을 주방향으로 가정
        except Exception:
            # OBB 계산 실패 시, 임의의 축을 주방향으로 설정 (에러 방지)
            main_axis_3d = np.array([1.0, 0.0, 0.0])

        # 주방향을 수평면(XY 평면)에 투영
        main_axis_2d = main_axis_3d[:2].copy()
        norm = np.linalg.norm(main_axis_2d)
        if norm > 1e-6:
            main_axis_2d /= norm
        else:
            main_axis_2d = np.array([1.0, 0.0]) # 주축이 거의 수직인 경우 x축을 기본값으로 사용

        # 객체 크기에 비례하여 카메라 거리 설정
        camera_distance = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 1.5
        
        max_attempts = 100 # 유효한 카메라 위치를 찾기 위한 최대 시도 횟수
        for _ in range(max_attempts):
            # 2. 제약 조건에 맞는 카메라 위치 샘플링
            # 수평면에서 무작위 방향 생성
            horizontal_angle_rad = np.random.uniform(0, 2 * np.pi)
            cam_direction_2d = np.array([np.cos(horizontal_angle_rad), np.sin(horizontal_angle_rad)])

            # 선수/선미(주방향) 20도 제외 조건 확인
            dot_product = np.abs(np.dot(main_axis_2d, cam_direction_2d))
            angle_diff_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

            if angle_diff_rad < np.deg2rad(self.exclusion_angle):
                continue # 제외 구역에 해당하면 다시 샘플링

            # 높이 20도 제한 조건 확인
            max_vertical_offset = camera_distance * np.tan(np.deg2rad(self.vertical_angle_limit))
            vertical_offset = np.random.uniform(-max_vertical_offset, max_vertical_offset)
            
            # 최종 카메라 위치 결정
            camera_pos = np.array([
                center[0] + camera_distance * cam_direction_2d[0],
                center[1] + camera_distance * cam_direction_2d[1],
                center[2] + vertical_offset
            ])

            # 3. HPR 수행
            _, visible_indices = pcd.hidden_point_removal(camera_pos, radius=camera_distance * 5)
            
            pcd_view = pcd.select_by_index(visible_indices)

            if len(pcd_view.points) > 100: # 유효한 뷰가 생성되었으면 루프 종료
                view_points_np = np.asarray(pcd_view.points)
                # 4. 최종 포인트 수로 리샘플링
                final_view_np = self.random_sample(view_points_np, self.hpr_points)
                return final_view_np
        
        # 만약 유효한 뷰를 찾지 못했다면, 원본에서 무작위 샘플링하여 반환 (Fallback)
        print_log(f'Warning: Could not find a valid HPR view after {max_attempts} attempts. Returning a random sample.', logger='Vessel')
        return self.random_sample(gt_pc, self.hpr_points)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        model_id = file_path.split('.')[0]

        # .npy 파일 로드 및 기본 전처리
        data = IO.get(os.path.join(self.pc_path, file_path)).astype(np.float32)
        data = self.random_sample(data, self.npoints)
        data = self.pc_norm(data)

        if self.is_finetuning:
            # 미세조정 시: (데이터, 레이블) 반환
            label = self.labels[idx]
            data_tensor = torch.from_numpy(data).float()
            return 'vessel', model_id, (data_tensor, label)
        else:
            # 사전학습 시: HPR을 적용하여 (partial_view, ground_truth) 쌍 반환
            # 'data'가 ground_truth가 됩니다.
            ground_truth_pc = data
            
            # HPR 뷰 생성
            partial_view_pc = self._generate_hpr_view(ground_truth_pc)
            
            # Tensor로 변환
            partial_view_tensor = torch.from_numpy(partial_view_pc).float()
            ground_truth_tensor = torch.from_numpy(ground_truth_pc).float()
            
            return 'vessel', model_id, (partial_view_tensor, ground_truth_tensor)

    def __len__(self):
        return len(self.file_list)