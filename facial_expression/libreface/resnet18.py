import torch
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(
        self,
        num_labels=8,
        dropout=0.1,
        fm_distillation=True):
        super().__init__()
        self.fm_distillation = fm_distillation
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
            nn.Sigmoid()  # keep because checkpoint was trained this way
        )
    def forward(self, images):
        features = self.encoder(images).flatten(1)
        labels = self.classifier(features)
        if self.fm_distillation:
            return labels, features
        return labels
