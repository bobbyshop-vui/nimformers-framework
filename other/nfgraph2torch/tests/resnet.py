"""ResNet-18 (same architecture as torchvision's) for conversion tests."""
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + (x if self.down is None else self.down(x)))


class ResNet18(nn.Module):
    def __init__(self, classes=1000, width=64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 7, 2, 3, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        layers, cin = [], w
        for i, cout in enumerate([w, 2 * w, 4 * w, 8 * w]):
            stride = 1 if i == 0 else 2
            layers += [Block(cin, cout, stride), Block(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        return self.fc(self.pool(self.layers(self.stem(x))).flatten(1))
