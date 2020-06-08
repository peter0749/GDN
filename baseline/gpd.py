import torch
import torch.nn as nn
import torch.nn.functional as F

class GPDClassifier(nn.Module):
    """
    Input: (batch_size, input_chann, 60, 60)
    """
    def __init__(self, input_chann, dropout=False, return_features=False):
        super(GPDClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_chann, 20, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.feature_dim = 500
        self.fc1 = nn.Linear(12*12*50, self.feature_dim)
        self.dp = nn.Dropout2d(p=0.5, inplace=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.feature_dim, 2)
        self.if_dropout = dropout
        self.return_features = return_features

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(-1, 7200)
        x = self.relu(self.fc1(x))
        f = x
        if self.if_dropout:
            x = self.dp(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)

        if self.return_features:
            return x, f
        return x
