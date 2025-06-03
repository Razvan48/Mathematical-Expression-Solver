import torch.nn as nn


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=16):
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu_0 = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU()
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_0 = nn.Flatten()

        self.fc_0 = nn.Linear(in_features=128 * 7 * 7, out_features=128)
        self.relu_3 = nn.ReLU()
        self.dropout_0 = nn.Dropout(p=0.4)
        self.fc_1 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, X):
        X = self.conv_0(X)
        X = self.relu_0(X)
        X = self.conv_1(X)
        X = self.relu_1(X)
        X = self.maxpool2d_0(X)

        X = self.conv_2(X)
        X = self.relu_2(X)
        X = self.maxpool2d_1(X)

        X = self.flatten_0(X)

        X = self.fc_0(X)
        X = self.relu_3(X)
        X = self.dropout_0(X)
        X = self.fc_1(X)

        return X


from_symbol_to_index = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '+': 10,
    '-': 11,
    '*': 12,
    '/': 13,
    '(': 14,
    ')': 15
}

from_index_to_symbol = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '+',
    11: '-',
    12: '*',
    13: '/',
    14: '(',
    15: ')'
}

