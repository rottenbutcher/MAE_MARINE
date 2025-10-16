import os
import random
import argparse

def create_split_files(data_dir, output_dir, train_ratio=0.8):
    """
    Scans a directory for .npy files and splits them into train.txt and test.txt.

    Args:
        data_dir (str): The directory containing the .npy files.
        output_dir (str): The directory where train.txt and test.txt will be saved.
        train_ratio (float): The proportion of the data to be used for training.
    """
    # Find all .npy files in the data directory
    try:
        npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if not npy_files:
            print(f"Error: No .npy files found in '{data_dir}'. Please check the path.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' does not exist.")
        return

    print(f"Found {len(npy_files)} total .npy files.")

    # Shuffle the file list randomly
    random.shuffle(npy_files)

    # Calculate the split index
    split_index = int(len(npy_files) * train_ratio)

    # Split the files into training and testing sets
    train_files = npy_files[:split_index]
    test_files = npy_files[split_index:]

    print(f"Splitting into {len(train_files)} training files and {len(test_files)} testing files.")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the training file list
    train_txt_path = os.path.join(output_dir, 'train.txt')
    with open(train_txt_path, 'w') as f:
        for file_name in train_files:
            f.write(f"{file_name}\n")
    print(f"Successfully created '{train_txt_path}'")

    # Write the testing file list
    test_txt_path = os.path.join(output_dir, 'test.txt')
    with open(test_txt_path, 'w') as f:
        for file_name in test_files:
            f.write(f"{file_name}\n")
    print(f"Successfully created '{test_txt_path}'")

    print("\nDone! You can now run your training script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test split files from a directory of .npy files.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the directory containing .npy point cloud files.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to the directory where train.txt and test.txt will be saved.")
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help="Ratio of training data (e.g., 0.8 for 80% train, 20% test).")

    args = parser.parse_args()

    create_split_files(args.data_dir, args.output_dir, args.split_ratio)




    
