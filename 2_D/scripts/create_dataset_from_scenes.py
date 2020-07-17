from typing import List, Optional
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import tqdm


def create_dataset_from_scenes(args) -> None:
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    dataset = create_dataset(args)
    scenes_path = os.path.join(os.getcwd(), args.input_dir, '*.*')
    for scene in tqdm.tqdm(glob.glob(scenes_path)):
        cap = cv2.VideoCapture(scene)
        frame_group_idx = 0
        while True:
            frame_groups = get_frame_groups(cap, args.num_frames, args.frame_rate)
            if frame_groups is None:
                break
            for frame_group in frame_groups:
                frame_group_paths = save_frame_group(frame_group, frame_group_idx, args.output_dir, scene)
                frame_group_idx += 1
                dataset = add_frame_group_paths(dataset, frame_group_paths, scene)
        cap.release()
    save_dataset(dataset, args.output_dir, args.dataset_file_name, args.split)


def create_dataset(args) -> pd.DataFrame:
    columns = [f'frame{int(idx)}' for idx in range(args.num_frames)]
    dataset = pd.DataFrame(columns=columns)
    return dataset


def get_frame_groups(video_capture: cv2.VideoCapture,
                     num_frames: int,
                     frame_rate: int) -> Optional[List[List[np.ndarray]]]:
    frame_groups = [list() for _ in range(frame_rate)]
    for idx in range(num_frames * frame_rate):
        ret, frame = video_capture.read()
        if not ret:
            return None
        frame_groups[int(idx % frame_rate)].append(frame)
    return frame_groups


def save_frame_group(frames: List[np.ndarray], frame_group_idx: int, output_dir: str, input_file: str) -> List[str]:
    filename = os.path.splitext(os.path.split(input_file)[1])[0]
    filename_to_frame = {f'{filename}__group_{frame_group_idx}__frame_{idx}.png': frame
                     for idx, frame in enumerate(frames)}
    filenames = []
    for filename, frame in filename_to_frame.items():
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame)
        filenames.append(filename)
    return filenames


def add_frame_group_paths(dataset: pd.DataFrame, frame_group_paths: List[str], scene: str) -> pd.DataFrame:
    row = {f'frame{idx}': frame_group_path for idx, frame_group_path in enumerate(frame_group_paths)}
    row['scene'] = scene
    dataframe = dataset.append(row, ignore_index=True)
    return dataframe


def save_dataset(dataset: pd.DataFrame, output_dir: str, dataset_file_name: str, split: float) -> None:
    num_groups = len(dataset)
    num_test = int(num_groups * split)
    test_indices = []

    for scene, indices in dataset.groupby(['scene']).indices.items():
        if len(test_indices) > num_test:
            break
        test_indices += list(indices)
    train_indices = set(dataset.index) - set(test_indices)

    dataset = dataset.drop('scene', axis=1)
    dataset_test = dataset.loc[test_indices]
    dataset_train = dataset.loc[train_indices]

    dataset_test.to_csv(os.path.join(output_dir, 'test_' + dataset_file_name), index=False)
    dataset_train.to_csv(os.path.join(output_dir, 'train_' + dataset_file_name), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir', '-i', type=str, required=True, help='Directory where input scene files are located'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, required=True, help='Directory where to save output frames'
    )
    parser.add_argument(
        '--dataset-file-name', '-dfm', type=str, required=True,
        help='Name of the output dataset csv file describing each group of frames'
    )
    parser.add_argument(
        '--num-frames', '-nf', type=int, required=True,
        help='Number of frames in a single group'
    )
    parser.add_argument(
        '--frame-rate', '-fr', type=int, required=True,
        help='Frequency at which frames are sampled'
    )
    parser.add_argument(
        '--split', '-s', type=float, required=True,
        help='Training/test dataset split'
    )

    args = parser.parse_args()
    create_dataset_from_scenes(args)
