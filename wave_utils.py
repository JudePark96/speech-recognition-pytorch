__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import numpy as np
import pandas as pd
import h5py

from scipy.io import wavfile
from tqdm import tqdm
from glob import glob


def data_loader(files):
    out = []
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)
    out = np.array(out)
    return out


def save_wav_hdf5(output_file):
    x_data = glob('./rsc/train/*.wav')
    x_data = data_loader(x_data)
    x_data = np.asarray([wav / wav.max() for wav in x_data])  # 최대값을 나누어 데이터 정규화
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)  # CNN 모델에 넣기 위한 데이터 shape 변경

    # 정답 값을 불러옵니다
    y_data = pd.read_csv('./rsc/train_answer.csv', index_col=0)
    # y_labels = y_data.columns.values
    y_data = y_data.values

    f = h5py.File(output_file, 'a')
    f['feature'] = x_data
    f['label'] = y_data
    f.close()


if __name__ == '__main__':
    # save_wav_hdf5('./rsc/train.hdf5')
    with h5py.File('./rsc/train.hdf5', 'r') as features_hdf:
        feature_keys = list(features_hdf.keys())
        a = features_hdf.get(feature_keys[0])
        b = features_hdf.get(feature_keys[1])
        print(a, b)
