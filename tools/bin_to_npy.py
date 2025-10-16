import os
import numpy as np
from tqdm import tqdm

# ------------------- 경로 설정 (이 부분만 수정하세요) ------------------- #

# 1. 원본 .bin 파일들이 있는 상위 폴더 경로를 지정해주세요.
# 예: source_root_folder = 'C:/Users/User/Desktop/my_vessels'
source_root_folder = '/home/jslee/Junseo/Vessel_Pointcloud/ShapeNet/Classified_Models/1'

# 2. 변환된 .npy 파일들을 저장할 폴더 경로를 지정해주세요.
# 예: destination_folder = 'C:/Users/User/Desktop/vessels_npy'
destination_folder = '/home/jslee/Junseo/Vessel_Pointcloud/ShapeNet/Classified_Models/Vessel'

# ------------------------------------------------------------------- #

# 저장할 폴더가 없으면 자동으로 생성합니다.
os.makedirs(destination_folder, exist_ok=True)

# 변환할 파일 목록을 저장할 리스트
target_filename = 'pts_xyz_8192.bin'
bin_file_paths = []

print(f"'{source_root_folder}' 폴더와 그 하위 모든 폴더에서 '{target_filename}' 파일을 검색합니다...")

# os.walk를 사용하여 모든 하위 폴더를 재귀적으로 탐색합니다.
for root, dirs, files in os.walk(source_root_folder):
    if target_filename in files:
        file_path = os.path.join(root, target_filename)
        bin_file_paths.append(file_path)

# 찾은 파일 경로들을 정렬하여 'vessel_1', 'vessel_2' 순서를 일관성 있게 만듭니다.
bin_file_paths.sort()

if not bin_file_paths:
    print(f"\n오류: '{target_filename}' 파일을 찾을 수 없습니다. 원본 폴더 경로를 확인해주세요.")
    exit()


# 파일 변환을 시작합니다.
print(f"총 {len(bin_file_paths)}개의 파일을 찾아 변환을 시작합니다.")

file_counter = 1
# tqdm을 사용하여 진행 상황을 시각적으로 보여줍니다.
for bin_file_path in tqdm(bin_file_paths, desc="변환 진행률"):
    try:
        # .bin 파일을 읽어옵니다. (데이터 타입: float32 가정)
        point_cloud = np.fromfile(bin_file_path, dtype=np.float32)
        
        # (N, 3) 형태로 재구성합니다.
        point_cloud = point_cloud.reshape(-1, 3)

        # 저장할 .npy 파일의 이름과 전체 경로를 설정합니다.
        npy_file_name = f'vessel_{file_counter}.npy'
        destination_file_path = os.path.join(destination_folder, npy_file_name)

        # .npy 파일로 저장합니다.
        np.save(destination_file_path, point_cloud)

        file_counter += 1

    except Exception as e:
        print(f"\n파일 처리 중 오류 발생: {bin_file_path}")
        print(f"오류 내용: {e}")

# 작업 완료 메시지를 출력합니다.
print(f"\n✨ 작업 완료! 총 {file_counter - 1}개의 파일을 '{destination_folder}' 폴더에 저장했습니다.")