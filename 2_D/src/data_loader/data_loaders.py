from src.data_loader.base import BaseDataLoader
from src.dataset import DD40Dataset, PretrainDataset


class DD40DataLoader(BaseDataLoader):
    """
    Duffy Duck 1940's Data Loader
    """
    def __init__(self, data_dir, batch_size, dataset_csv, shuffle=True, return_paths=False,
                 validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = DD40Dataset(self.data_dir, train=training, dataset_file=dataset_csv, return_paths=return_paths)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class PretrainDataLoader(BaseDataLoader):
    """
    PretrainDataLoader Data Loader
    """
    def __init__(self, data_dir, batch_size, dataset_csv, min_zoom, max_zoom, min_crop_size, max_crop_size,
                 min_translation, shuffle=True, return_paths=False, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = PretrainDataset(self.data_dir, train=training, dataset_file=dataset_csv,
                                       return_paths=return_paths, min_zoom=min_zoom, max_zoom=max_zoom,
                                       min_crop_size=min_crop_size, max_crop_size=max_crop_size,
                                       min_translation=min_translation)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
