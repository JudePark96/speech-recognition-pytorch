__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch
from torch.utils.data import DataLoader

from data_loader import SpeechTestDataset, SpeechDataset
from models.model import BaselineModel
from models.tdnn import TDNN

if __name__ == '__main__':
    model = TDNN()
    model.load_state_dict(torch.load('./rsc/output_tdnn/pt_tdnn_0.bin'))
    train_dataset = SpeechDataset('./rsc/train_features.hdf5', './rsc/train_answer.csv')
    test_dataset = SpeechTestDataset('./rsc/test_features.hdf5')
    # print(model(torch.randn(32, 80, 101)))
    train_dataloader = DataLoader(train_dataset, batch_size=64,
                                  num_workers=16, pin_memory=True)

    for batch in train_dataloader:
        print([float(i) for i in model(batch['feature'])[0]])
        break
    pass