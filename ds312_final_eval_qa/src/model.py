import torch
from torch import nn


class CompactCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        #Extract image features using three convolutional blocks
        self.features = nn.Sequential(
            self._make_block(3, 32),
            self._make_block(32, 64),
            self._make_block(64, 128),
        )

        #Reduce each feature map to one value
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        #Convert the learned features into class scores
        self.classifier = nn.Linear(128, num_classes)

    @staticmethod
    def _make_block(
        in_channels: int,
        out_channels: int,
    ) -> nn.Sequential:
        #Use two convolutions followed by spatial downsampling
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        #Extract features and reduce their spatial dimensions
        features = self.features(images)
        features = self.pool(features)

        #Flatten the features and calculate class scores
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)
