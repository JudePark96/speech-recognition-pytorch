__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import h5py
from torch.utils.data import Dataset, DataLoader

import torch
import pandas as pd


class SpeechDataset(Dataset):
    def __init__(self, file_path, answer_path) -> None:
        self.file_path = file_path
        self.labels = torch.Tensor(pd.read_csv(answer_path, index_col=0).values)

        with h5py.File(file_path, 'r') as features_hdf:
            self.feature_keys = list(features_hdf.keys())
            self.num_instances = features_hdf.get(self.feature_keys[0]).shape[0]

        print(f'total instances is {self.num_instances}')

    def __getitem__(self, index: int) -> dict:
        features = self._read_hdf_features(index)
        return features

    def __len__(self) -> int:
        return self.num_instances

    def _read_hdf_features(self, index):
        features = {}

        with h5py.File(self.file_path, 'r') as features_hdf:
            features['feature'] = features_hdf['feature'][index]

        features['label'] = self.labels[index]

        return features


def get_data_loader(dataset, bs, shuffle, num_workers, pin_memory):
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
