import os
import random
import argparse
import numpy as np

def create_3way_split_files(data_dir, output_dir, ratios=(0.6, 0.2, 0.2), add_labels=False, num_classes=10):
    """
    Scans a directory for .npy files and splits them into pretrain.txt, 
    finetune_train.txt, and finetune_test.txt.
    Optionally adds dummy labels for fine-tuning files.
    """
    try:
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if not npy_files:
            print(f"Error: No .npy files found in '{data_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' does not exist.")
        return

    print(f"Found {len(npy_files)} total .npy files.")
    random.shuffle(npy_files)

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Ratios must sum to 1.")

    pretrain_idx = int(len(npy_files) * ratios[0])
    finetune_train_idx = pretrain_idx + int(len(npy_files) * ratios[1])

    pretrain_files = npy_files[:pretrain_idx]
    finetune_train_files = npy_files[pretrain_idx:finetune_train_idx]
    finetune_test_files = npy_files[finetune_train_idx:]

    print(f"Splitting into {len(pretrain_files)} pre-train, {len(finetune_train_files)} fine-tune train, and {len(finetune_test_files)} fine-tune test files.")
    os.makedirs(output_dir, exist_ok=True)

    # 1. pretrain.txt (레이블 없음)
    with open(os.path.join(output_dir, 'pretrain.txt'), 'w') as f:
        for file_name in pretrain_files:
            f.write(f"{file_name}\n")
    print(f"Successfully created 'pretrain.txt'")

    # 2. finetune_train.txt (레이블 포함/미포함)
    with open(os.path.join(output_dir, 'finetune_train.txt'), 'w') as f:
        for file_name in finetune_train_files:
            if add_labels:
                # This assumes filename format 'vessel_CLASSID_....npy' or similar
                # For now, we add random dummy labels for demonstration
                label = random.randint(0, num_classes - 1)
                f.write(f"{file_name} {label}\n")
            else:
                f.write(f"{file_name}\n")
    print(f"Successfully created 'finetune_train.txt'")
    
    # 3. finetune_test.txt (레이블 포함/미포함)
    with open(os.path.join(output_dir, 'finetune_test.txt'), 'w') as f:
        for file_name in finetune_test_files:
            if add_labels:
                label = random.randint(0, num_classes - 1)
                f.write(f"{file_name} {label}\n")
            else:
                f.write(f"{file_name}\n")
    print(f"Successfully created 'finetune_test.txt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 3-way data split files.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory with .npy files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the .txt split files.")
    # 실제 데이터에 레이블이 있다면 add_labels 플래그를 사용하고, Vessel_Dataset.py에서 레이블을 파싱하도록 수정해야 합니다.
    # 지금은 레이블 없는 파일만 생성합니다.
    args = parser.parse_args()
    create_3way_split_files(args.data_dir, args.output_dir)