"""One-dimensional CNN used by the CC method."""

from torch import nn


class FeatureExtractor(nn.Module):
    """Convert a ``[batch, 1, 1024]`` signal into a 512-D feature."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=15),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )

    def forward(self, inputs):
        features = self.network(inputs)
        return features.flatten(start_dim=1)
