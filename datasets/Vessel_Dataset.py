import os
import torch
import numpy as np
import torch.utils.data as data
import open3d as o3d
from .io import IO
from .build import DATASETS
from utils.logger import print_log

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

@DATASETS.register_module()
class Vessel(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        self.hpr_points = config.get('HPR_POINTS', 0) # HPR 포인트 수, 없으면 0
        self.vertical_angle_limit = config.get('HPR_VERTICAL_ANGLE', 20)
        self.exclusion_angle = config.get('HPR_EXCLUSION_ANGLE', 20)
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        
        print_log(f'[DATASET] Loading data from {self.data_list_file}', logger='Vessel')
        
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = [line.strip().split()[0] for line in lines if line.strip()]
        self.labels = [int(line.strip().split()[1]) for line in lines if len(line.strip().split()) > 1]

        self.is_finetuning = len(self.labels) == len(self.file_list) and len(self.labels) > 0
        
        if self.is_finetuning:
             print_log(f'[DATASET] Fine-tuning mode enabled. {len(self.labels)} labels loaded.', logger='Vessel')
        else:
             print_log(f'[DATASET] Pre-training mode enabled.', logger='Vessel')
             if self.hpr_points > 0:
                 print_log(f'[DATASET] Generating HPR views with {self.hpr_points} points.', logger='Vessel')
             else:
                 print_log(f'[DATASET] Loading full point clouds for Autoencoder training (e.g., dVAE).', logger='Vessel')

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded for {self.subset}.', logger='Vessel')

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def random_sample(self, pc, num):
        if pc.shape[0] < num:
            indices = np.random.choice(pc.shape[0], num, replace=True)
        else:
            indices = np.random.choice(pc.shape[0], num, replace=False)
        return pc[indices]
    
    def _generate_hpr_view(self, gt_pc):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_pc)
        center = pcd.get_center()

        try:
            oriented_bbox = pcd.get_oriented_bounding_box()
            main_axis_3d = oriented_bbox.R[:, 0]
        except Exception:
            main_axis_3d = np.array([1.0, 0.0, 0.0])

        main_axis_2d = main_axis_3d[:2].copy()
        norm = np.linalg.norm(main_axis_2d)
        main_axis_2d = main_axis_2d / norm if norm > 1e-6 else np.array([1.0, 0.0])

        camera_distance = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 1.5
        
        for _ in range(100):
            horizontal_angle_rad = np.random.uniform(0, 2 * np.pi)
            cam_direction_2d = np.array([np.cos(horizontal_angle_rad), np.sin(horizontal_angle_rad)])
            
            angle_diff_rad = np.arccos(np.clip(np.abs(np.dot(main_axis_2d, cam_direction_2d)), -1.0, 1.0))
            if angle_diff_rad < np.deg2rad(self.exclusion_angle):
                continue

            max_vertical_offset = camera_distance * np.tan(np.deg2rad(self.vertical_angle_limit))
            vertical_offset = np.random.uniform(-max_vertical_offset, max_vertical_offset)
            
            camera_pos = np.array([
                center[0] + camera_distance * cam_direction_2d[0],
                center[1] + camera_distance * cam_direction_2d[1],
                center[2] + vertical_offset
            ])

            _, visible_indices = pcd.hidden_point_removal(camera_pos, radius=camera_distance * 5)
            
            pcd_view = pcd.select_by_index(visible_indices)
            if len(pcd_view.points) > 100:
                return self.random_sample(np.asarray(pcd_view.points), self.hpr_points)
        
        return self.random_sample(gt_pc, self.hpr_points)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        model_id = file_path.split('.')[0]
        data = IO.get(os.path.join(self.pc_path, file_path)).astype(np.float32)
        data = self.random_sample(data, self.npoints)
        ground_truth_pc = self.pc_norm(data)

        if self.is_finetuning:
            label = self.labels[idx]
            data_tensor = torch.from_numpy(ground_truth_pc).float()
            return 'vessel', model_id, (data_tensor, label)
        else:
            # 사전학습 모드 분기 처리
            if self.hpr_points > 0:
                # HPR 모드 (MAE, M2AE 용): (partial, ground_truth) 쌍 반환
                partial_view_pc = self._generate_hpr_view(ground_truth_pc)
                partial_view_tensor = torch.from_numpy(partial_view_pc).float()
                ground_truth_tensor = torch.from_numpy(ground_truth_pc).float()
                return 'vessel', model_id, (partial_view_tensor, ground_truth_tensor)
            else:
                # Autoencoder 모드 (dVAE 용): ground_truth 텐서 하나만 반환
                ground_truth_tensor = torch.from_numpy(ground_truth_pc).float()
                return 'vessel', model_id, ground_truth_tensor

    def __len__(self):
        return len(self.file_list)