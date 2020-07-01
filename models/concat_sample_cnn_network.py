__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch
import torch.nn as nn
import torch.nn.functional as F


from models.conv_block import ConvBlock

class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()

        self.conv_seq_1 = nn.Sequential(
            ConvBlock('gelu', 16000, 128, 3, 3, 4, 0.15),
            ConvBlock('gelu', 128, 128, 3, 3, 4, 0.15),
            ConvBlock('gelu', 128, 128, 3, 3, 4, 0.15),
            ConvBlock('gelu', 128, 128, 3, 3, 4, 0.15),
            ConvBlock('gelu', 128, 128, 3, 3, 4, 0.15)
        )

        self.conv_seq_2 = nn.Sequential(
            ConvBlock('gelu', 16000, 128, 5, 3, 8, 0.15),
            ConvBlock('gelu', 128, 128, 5, 3, 8, 0.15),
            ConvBlock('gelu', 128, 128, 5, 3, 8, 0.15),
            ConvBlock('gelu', 128, 128, 5, 3, 8, 0.15),
            ConvBlock('gelu', 128, 128, 5, 3, 8, 0.15)
        )
        
        self.conv_seq_3 = nn.Sequential(
            ConvBlock('gelu', 16000, 128, 7, 3, 12, 0.15),
            ConvBlock('gelu', 128, 128, 7, 3, 12, 0.15),
            ConvBlock('gelu', 128, 128, 7, 3, 12, 0.15),
            ConvBlock('gelu', 128, 128, 7, 3, 12, 0.15),
            ConvBlock('gelu', 128, 128, 7, 3, 12, 0.15)
        )

        self.classifier = nn.Sequential(nn.Linear(in_features=384, out_features=192),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=192, out_features=96),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=96, out_features=30))

        self.apply(self._init_weights)


    def forward(self, data):
        seq1, seq2, seq3 = self.conv_seq_1(data), \
                           self.conv_seq_2(data), \
                           self.conv_seq_3(data)

        # [bs x (128*3) x 1]
        features = torch.cat([seq1, seq2, seq3], dim=1)
        # [bs x (128*3)]
        features = features.squeeze(dim=-1)

        logits = self.classifier(features)

        return F.log_softmax(logits, dim=1)

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)


if __name__ == '__main__':
    # [n x n_mels x time]
    data = torch.randn(32, 16000, 1)
    cnn = SampleCNN()

    print(cnn(data).shape)
    print(cnn(data))
