__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, act_fn, in_channel, out_channel, kernel_size, stride, padding, dropout):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool1d(kernel_size, stride=3)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.sequential = nn.Sequential(
            self.conv,
            self.bn,
            self.act,
            self.dropout,
            self.max_pool
        )

    def forward(self, data):
        return self.sequential(data)

if __name__ == '__main__':
    # [n x n_mels x time]
    data = torch.randn(32, 16000, 1)
    conv1 = ConvBlock('relu', 16000, 128, 7, 3, 12, 0.15)
    conv2 = ConvBlock('relu', 128, 128, 7, 3, 12, 0.15)
    conv3 = ConvBlock('relu', 128, 128, 7, 3, 12, 0.15)
    conv4 = ConvBlock('relu', 128, 128, 7, 3, 12, 0.15)
    print(conv4(conv3(conv2(conv1(data)))).shape)
