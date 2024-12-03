import time

import torch
from torch.nn import Module, Conv2d, MaxPool2d, Dropout, Dropout2d, Mish, Linear, Sequential, AdaptiveMaxPool2d, Flatten
from torchsummary import summary


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super().__init__()
        self.depth_wise = Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.point_wise = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.activation = Mish()
        self.dropout = Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FCBlock(Module):
    def __init__(self, in_features, out_features, dropout_prob=0.4):
        super().__init__()
        self.linear = Linear(in_features, out_features)
        self.activation = Mish()
        self.dropout = Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Nanonet(Module):
    def __init__(self, color_channels=3, classes=10, mid_layer_depth=5, base_index=6):
        super().__init__()

        features = []
        classifier = []

        for i in range(base_index, base_index + mid_layer_depth // 2):
            in_ch = 2 ** i
            out_ch = 2 ** (i + 1)
            features.append(ConvBlock(in_ch, out_ch))

        for i in reversed(range(base_index, base_index + mid_layer_depth // 2)):
            in_ch = 2 ** (i + 1)
            out_ch = 2 ** i
            classifier.append(FCBlock(in_ch, out_ch))

        self.features = Sequential(
            ConvBlock(color_channels, 2 ** base_index),
            *features,
        )

        self.feature_expand = Sequential(
            AdaptiveMaxPool2d(1),
            Flatten()
        )
        self.classifier = Sequential(
            *classifier,
            FCBlock(2 ** base_index, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.feature_expand(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    image_size = 448
    model = Nanonet()
    print("--- model info ---")
    print(summary(model, torch.randn(3, image_size, image_size).shape))

    print("--- model test ---")
    model.eval()
    noise = torch.randn(1, 3, image_size, image_size)
    start_time = time.time()
    print(f"input: {noise.shape}")
    print(f"output: {model(noise).shape}")
    end_time = time.time()
    print(f"predict time: {end_time - start_time:.4f}s")
