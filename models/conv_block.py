__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, act_fn, in_channel, out_channel, kernel_size):
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.GELU() if act_fn == 'gelu' else nn.ReLU()

    def forward(self, data):
        return self.max_pool(self.act(self.bn(self.conv(data))))



if __name__ == '__main__':
    # [n x n_mels x time]
    data = torch.randn(32, 80, 101)
    conv = ConvBlock('relu', 80, 80, 10)
    print(conv(data).shape)
