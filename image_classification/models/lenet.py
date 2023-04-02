import torch
from torch.nn import Conv2d, Linear, MaxPool2d, Module, ReLU


class LeNetBackbone(Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.act = ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        return x
    

class LeNetHead(Module):
    def __init__(self, num_classes, input_features=16 * 5 * 5):
        super().__init__()
        self.fc1 = Linear(in_features=input_features, out_features=120)
        self.fc2 = Linear(in_features=120, out_features=84)
        self.fc3 = Linear(in_features=84, out_features=num_classes)
        self.act = ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        self.backbone = LeNetBackbone(input_channels=input_channels)
        self.head = LeNetHead(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
