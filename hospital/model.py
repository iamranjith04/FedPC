import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.encoder(x).flatten(1)
        out = self.fc(feat)
        return (out, feat) if return_feat else out
