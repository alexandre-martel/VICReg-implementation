import torch
import torch.nn as nn
from torchvision import models

class VICRegModel(nn.Module):
    def __init__(self, embedding_dim=512, projection_dim=2048):
        super().__init__()

        # BACKBONE (ResNet18 without final classification layer)
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity() # Remove final classification layer to get embeddings
        # NOTE : We shouldn't do VICReg directly on final embeddings from ResNet18 output layer because it is too small (512 dimensions) 
        # and can't capture enough information. Hence, we add a projector MLP to increase the dimension to 2048.

        # PROJECTOR (Simple MLP here)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim) 
        )

    def forward(self, x):
        h = self.backbone(x) # Features (512 dims)
        z = self.projector(h) # Projections (2048 dims)
        return z