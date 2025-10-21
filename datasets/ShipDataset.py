# /datasets/ShipDataset.py

import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
from .io import IO

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

@DATASETS.register_module()
class SimulationShip(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.npoints = config.npoints
        
        # Simulation_Ship 폴더 내의 모든 .npy 파일을 읽어옵니다.
        self.file_list = sorted([f for f in os.listdir(self.data_root) if f.endswith('.npy')])
        
        # 파일 이름 순서를 기준으로 0부터 699까지의 고유한 레이블을 생성합니다.
        self.labels = {file_name: i for i, file_name in enumerate(self.file_list)}
        
        print(f'[DATASET] SimulationShip: {len(self.file_list)} files loaded.')
        # 점 개수가 npoints보다 많을 경우, 랜덤 샘플링을 위한 인덱스
        self.permutation = np.arange(self.npoints)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_root, file_name)
        
        points = np.load(file_path).astype(np.float32)

        # npoints보다 많으면 샘플링, 적으면 그대로 사용
        num_points_in_file = points.shape[0]
        if num_points_in_file > self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=False)
            points = points[indices]
        elif num_points_in_file < self.npoints:
            indices = np.random.choice(num_points_in_file, self.npoints, replace=True)
            points = points[indices]

        points = pc_normalize(points)
        points = torch.from_numpy(points).float()
        
        label = self.labels[file_name]
        return 'simulation_ship', os.path.splitext(file_name)[0], (points, label)

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class RealShip(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.subset = config.subset
        
        # 폴더 이름('target1', 'FP_target14' 등)을 기준으로 클래스-인덱스 맵을 자동으로 생성합니다.
        self.class_map = {d: i for i, d in enumerate(sorted([f for f in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, f))]))}

        list_file = os.path.join(os.path.dirname(self.root), f'real_ship_{self.subset}.txt')
        
        with open(list_file, 'r') as f:
            self.file_list = [line.strip().replace('/', os.path.sep) for line in f.readlines()]
            
        print(f'[DATASET] RealShip({self.subset}): {len(self.file_list)} files loaded.')
        print(f'Class mapping: {self.class_map}')


    def __getitem__(self, index):
        relative_path = self.file_list[index]
        
        # 파일 경로에서 클래스 이름(폴더명)을 추출하여 레이블로 사용
        class_name = os.path.basename(os.path.dirname(relative_path))
        label = self.class_map[class_name]

        full_path = os.path.join(self.root, relative_path)
        structured_array = np.load(full_path)
        points = np.vstack([structured_array['x'], structured_array['y'], structured_array['z']]).T.astype(np.float32)
        
        # 샘플링 및 정규화
        num_points_in_file = points.shape[0]
        if num_points_in_file > self.npoints:
            # 점이 많으면 npoints만큼 무작위 샘플링
            indices = np.random.choice(num_points_in_file, self.npoints, replace=False)
            current_points = points[indices]
        elif num_points_in_file < self.npoints:
            # 점이 부족하면 중복을 허용하여 npoints만큼 샘플링
            indices = np.random.choice(num_points_in_file, self.npoints, replace=True)
            current_points = points[indices]
        else:
            # 점 개수가 정확히 맞으면 그대로 사용
            current_points = points
        
        return 'RealShip', 'sample', (current_points, label)

    def __len__(self):
        return len(self.file_list)