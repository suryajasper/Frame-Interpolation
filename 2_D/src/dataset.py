import os
import random

from torchvision import transforms
from PIL import Image
from skimage import io
from torch.utils import data
import torch
import numpy as np
import pandas as pd


class Tile(object):

    def __init__(self, reps):
        self.reps = reps

    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = np.expand_dims(sample, axis=2)
        sample  = np.tile(sample, self.reps)
        return Image.fromarray(sample)


class DD40Dataset(data.Dataset):

    def __init__(self, directory, dataset_file, train, return_paths=False):
        self.train = train
        self.return_paths = return_paths
        self.directory = directory
        self.transform = transforms.Compose([
  #          transforms.Grayscale(),
   #         Tile(reps=(1, 1, 3)),
            transforms.Resize(size=(196, 196)),
            transforms.ToTensor(),
        ])
        self.dataset_descriptor = pd.read_csv(os.path.join(directory, dataset_file))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        for i in range(5):
            frame_name = 'frame' + str(i)
            frame_path = os.path.join(self.directory, self.dataset_descriptor.iloc[idx][frame_name])
            frame = Image.open(frame_path)
            sample[frame_name] = self.transform(frame)
            if self.return_paths:
                sample[frame_name + '_path'] = self.dataset_descriptor.iloc[idx][frame_name]
            
        return sample

    def __len__(self):
        return len(self.dataset_descriptor.index)


class PretrainDataset(data.Dataset):

    def __init__(self, directory, dataset_file, train, min_zoom, max_zoom,
                 min_crop_size, max_crop_size, min_translation, return_paths=False):
        self.train = train
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.min_translation = min_translation
        self.return_paths = return_paths
        self.directory = directory
        self.transform = transforms.Compose([
            transforms.Resize(size=(196, 196)),
            transforms.ToTensor(),
        ])
        self.dataset_descriptor = pd.read_csv(os.path.join(directory, dataset_file))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_name = 'frame' + str(0)
        frame_path = os.path.join(self.directory, self.dataset_descriptor.iloc[idx][frame_name])
        frame = Image.open(frame_path)
        frames = self._get_augmented_frames(frame)

        sample = {}
        for i, frame in enumerate(frames):
            frame_name = 'frame' + str(i)
            sample[frame_name] = self.transform(frame)
            if self.return_paths:
                sample[frame_name + '_path'] = self.dataset_descriptor.iloc[idx][frame_name]

        return sample

    def __len__(self):
        return len(self.dataset_descriptor.index)

    def _get_augmented_frames(self, image):
        transformation = np.random.choice(['zoom', 'translate'])
        if transformation == 'zoom':
            return self._zoom(image)
        return self._translate(image)

    def _zoom(self, image):
        lf_factor = random.uniform(self.min_zoom, self.max_zoom)
        rt_factor = random.uniform(1.0 - self.max_zoom, 1.0 - self.min_zoom)
        tp_factor = random.uniform(self.min_zoom, self.max_zoom)
        bt_factor = random.uniform(1.0 - self.max_zoom, 1.0 - self.min_zoom)

        frames = [image]
        for i in range(2):
            frame = frames[i]
            w, h = frame.size
            lf = lf_factor * w
            rt = rt_factor * w
            tp = tp_factor * h
            bt = bt_factor * h
            new_frame = self._crop_resize(frame, lf, rt, tp, bt)
            frames.append(new_frame)

        return frames

    def _translate(self, image):
        w, h = image.size
        direction = np.random.choice(['vertical', 'horizontal'])
        crop_size = random.uniform(self.min_crop_size, self.max_crop_size)
        translation = random.uniform(self.min_translation, (1 - crop_size) / 2)

        frames = []
        for i in range(3):
            frame = image

            if direction == 'horizontal':
                lf = (w * (1 - crop_size)  / 2.0) + (i - 1) * translation * w
                rt = lf + crop_size * w
                tp = 0.0 * h
                bt = 1.0 * h

            else:
                lf = 0.0 * w
                rt = 1.0 * w
                tp = (h * (1 - crop_size)  / 2.0) + (i - 1) * translation * h
                bt = tp + crop_size * h

            new_frame = self._crop_resize(frame, lf, rt, tp, bt)
            frames.append(new_frame)

        return frames

    def _crop_resize(self, image, lf, rt, tp, bt):
        w, h = image.size
        image = image.crop((lf, tp, rt, bt))
        image = image.resize((w, h))
        return image
