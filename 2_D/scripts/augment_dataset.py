from typing import List
from uuid import uuid4
import argparse
import os
import sys

sys.path.append('..')

from Augmentor.Operations import Operation
from PIL import Image
import Augmentor
import numpy as np
import pandas as pd
import tqdm

from src.utils.utils import batch


TripletPaths = List[List[str]]
TripletImages = List[List[np.ndarray]]


class ResizeKeepingRatio(Operation):

    def __init__(self, probability: float, dimension: int):
        Operation.__init__(self, probability)
        self.dim = dimension

    def perform_operation(self, triplet: Image) -> Image:
        width, height = triplet[0].size
        if width > height:
            new_size = int((self.dim / height) * width), self.dim
        else:
            new_size = self.dim, int((self.dim / width)) * height
        return [image.resize(new_size) for image in triplet]


def augment_dataset(options):
    df = pd.DataFrame()
    all_triplets = get_triplets(options.input_dataset_path)
    if not os.path.exists(options.output_dir):
        os.mkdir(options.output_dir)
    for triplet_batch in tqdm.tqdm(batch(all_triplets)):
        images = get_images(triplet_batch)
        augmented_images = augment_images(images, options.output_dimension, options.augmentation_multiplier)
        triplet_paths = save_images(options.output_dir, augmented_images)
        df = add_triplet_paths_to_dataframe(triplet_paths, df)
    df.to_csv(os.path.join(options.output_dir, options.output_dataset))


def get_triplets(dataset_path: str) -> TripletPaths:
    df = pd.read_csv(dataset_path)
    triplets = [[df.iloc[idx][col] for col in df.columns]
                for idx, row in df.iterrows()]
    return triplets


def get_images(triplets_paths: TripletPaths) -> TripletImages:
    return [[np.asarray(Image.open(image_path)) for image_path in triplet]
            for triplet in triplets_paths]


def augment_images(images: TripletImages, output_dimension: int, augmentation_multiplier: int) -> TripletImages:
    p = Augmentor.DataPipeline(images)
    p = apply_augmentation_pipeline(p, output_dimension)
    augmented_images = p.sample(int(len(images) * augmentation_multiplier))
    return augmented_images


def apply_augmentation_pipeline(p: Augmentor.DataPipeline, output_dimension: int) -> Augmentor.DataPipeline:
    p.random_color(1.0, min_factor=0.0, max_factor=0.0)
    p.rotate(1.0, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(0.5)
    p.skew(0.2, magnitude=0.5)
    p.add_operation(ResizeKeepingRatio(1.0, dimension=output_dimension))
    p.crop_by_size(1.0, width=output_dimension, height=output_dimension, centre=False)
    return p


def save_images(output_dir: str, triplet_images: TripletImages) -> TripletPaths:
    triplet_paths = [list() for _ in triplet_images]
    for i, triplet in enumerate(triplet_images):
        triplet_id = uuid4()
        for j, image in enumerate(triplet):
            image = Image.fromarray(image)
            filename = f'{triplet_id}_{i}_{j}.png'
            path = os.path.join(output_dir, filename)
            image.save(path)
            triplet_paths[i].append(filename)
    return triplet_paths


def add_triplet_paths_to_dataframe(triplet_paths: TripletPaths, dataframe: pd.DataFrame) -> pd.DataFrame:
    for triplet in triplet_paths:
        row = {f'frame{idx}': path for idx, path in enumerate(triplet)}
        dataframe = dataframe.append(row, ignore_index=True)
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dataset-path', '-idp', type=str, required=True,
        help='Name of the input dataset csv file describing each group of frames'
    )
    parser.add_argument(
        '--output-dataset', '-od', type=str, required=True,
        help='Name of the output dataset csv file describing each group of frames'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, required=True, help='Directory where to save output frames'
    )
    parser.add_argument(
        '--output-dimension', '-dim', type=int, required=True, help='Dimension of output frames'
    )
    parser.add_argument(
        '--augmentation-multiplier', '-am', type=int, required=True, default=4,
        help='By how many times to increase the size of the dataset'
    )

    args = parser.parse_args()
    augment_dataset(args)
