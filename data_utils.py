__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from tqdm import tqdm
from typing import Sequence

import os
import h5py
import torch
import torchaudio
import torch.nn as nn
import torchaudio.transforms as AT


def get_mel_spectogram(file_path, n_fft, win_length, hop_length, n_mels) -> torch.Tensor:
    x, sr = torchaudio.load(file_path, normalization=lambda x: torch.abs(x).max())
    mel_spectrogram = nn.Sequential(
        AT.MelSpectrogram(sample_rate=sr,
                          n_fft=n_fft,
                          win_length=win_length,
                          hop_length=hop_length,
                          n_mels=n_mels),
        AT.AmplitudeToDB()
    )

    return mel_spectrogram(x)


def get_audio_dataset(folder_path, output_file, n_fft, win_length, hop_length, n_mels) -> torch.Tensor:
    if folder_path[-1] != '/':
        raise ValueError("folder path should have end with '/'.")

    audio_files = os.listdir(folder_path)
    features = []

    for file in tqdm(audio_files):
        features.append(get_mel_spectogram(file_path=folder_path + file,
                                           n_fft=n_fft,
                                           win_length=win_length,
                                           hop_length=hop_length,
                                           n_mels=n_mels).numpy())

    # [n x 1 x n_mels x time]
    features = torch.Tensor(features)

    # [n x n_mels x time]
    features = features.squeeze(dim=1)

    # h5py 로 features 를 추출.
    f = h5py.File(output_file, 'a')
    f['feature'] = features
    f.close()

    return features


def read_hdf5(file_path) -> torch.Tensor:
    # f = h5py.File(file_path, 'r')
    # print("Keys: %s" % f.keys())
    # feature_key = list(f.keys())[0]
    # data = f[feature_key]

    with h5py.File(file_path, 'r') as features_hdf:
        feature_keys = list(features_hdf.keys())
        print(feature_keys)
        num_instances = features_hdf.get(feature_keys[0])
        for i in num_instances:
            print(i)
            break
        print(num_instances)


if __name__ == '__main__':
    read_hdf5('rsc/train_features.hdf5')

