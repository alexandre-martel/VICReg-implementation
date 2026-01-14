import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.TwoViewTransformClass import TwoViewTransform
from src.VICREgModelClass import VICRegModel

base_transform = transforms.Compose([
    transforms.RandomResizedCrop(32), # CIFAR = 32x32 pixels images
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])