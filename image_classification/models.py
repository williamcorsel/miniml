import torch
from torch.nn import Module, Conv2d, Linear, MaxPool2d
from torch.nn.functional import relu


class LeNet(Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        self.conv1 = Conv2d(input_channels, 6, 5) # in_channels, out_channels, kernel_size
        self.pool2 = MaxPool2d(2, 2) # kernel_size, stride
        self.conv3 = Conv2d(6, 16, 5)
        self.pool4 = MaxPool2d(2, 2)
        self.fc5 = Linear(16 * 5 * 5, 120)
        self.fc6 = Linear(120, 84)
        self.fc7 = Linear(84, num_classes)

    def forward(self, x):
        x = self.pool2(relu(self.conv1(x)))
        x = self.pool4(relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = relu(self.fc5(x))
        x = relu(self.fc6(x))
        x = self.fc7(x)
        return x


