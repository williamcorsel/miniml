import torch
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, BatchNorm2d, Sequential, AdaptiveAvgPool2d
import torch.nn.functional as F

class ResNetBlock(Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # Layer 1
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)

        # Layer 2
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)

        self.act = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # save identity for skip connection
        identity = x

        # Layers
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection before activation
        out += identity
        out = self.act(out)

        return out

class ResNetBackbone(Module):
    def __init__(self, input_channels=3):
        super().__init__()

        # Initial layer
        self.conv1 = Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        # Downsample if necessary
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

        # List of blocks in this layer
        layers = [ResNetBlock(inplanes, planes, stride, downsample)]
        layers += [ResNetBlock(planes, planes) for _ in range(1, blocks)]

        # Return as sequential container
        return Sequential(*layers)
    
    def forward(self, x):
        # Initial layer
        x = self.relu(self.bn1(self.conv1(x)))

        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNetHead(Module):
    def __init__(self, num_classes, input_features=512):
        super().__init__()
        # Adaptive pool to reduce features regardless of input size
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(input_features, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNet(Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        self.backbone = ResNetBackbone(input_channels=input_channels)
        self.head = ResNetHead(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x