from torch.nn import Module, Sequential, Conv2d, MaxPool2d, ReLU


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            MaxPool2d(2, 2),
            ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MnistCnn(Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential()
        self.feature_expand = Sequential()
        self.classifier = Sequential()

    def forward(self, x):
        return x
