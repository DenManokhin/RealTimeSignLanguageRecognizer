import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=25)
        self.dropout50 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # output 28x28x32
        x = self.pool(x)  # output 14x14x32

        x = F.relu(self.conv2(x))  # output 10x10x64
        x = self.pool(x)  # output 5x5x64

        x = self.dropout50(x)

        x = F.relu(self.conv3(x))  # output 1x1x128

        x = x.squeeze()  # 1x1xfeatures -> features
        x = F.relu(self.fc1(x))

        x = self.dropout50(x)

        # if batch_size=1 then x.size=1, so need to add a dimension
        if len(x.size()) == 1:
            x = x.unsqueeze(0)

        x = F.softmax(self.fc2(x), dim=1)

        return x
