import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


from src.TwoViewTransformClass import TwoViewsTransform
from src.VICRegModelClass import VICRegModel
from src.VICRegLoss import vicreg_loss
from src.evaluate import evaluate

# DATA DOWNLOAD AND PREPARATION

# We define small transformations for CIFAR in order to create two different (but similar) views of the same image.
base_transform = transforms.Compose([
    transforms.RandomResizedCrop(32), # CIFAR = 32x32 pixels images
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset_vicreg = torchvision.datasets.CIFAR10( # Without label because VICReg is self-supervised
    root='./data', train=True, download=True, 
    transform=TwoViewsTransform(base_transform)
)

test_dataset = torchvision.datasets.CIFAR10( # With labels for evaluation
    root='./data', train=False, download=True, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)

train_dataset_labels = torchvision.datasets.CIFAR10( # SMall classifier to evaluate learned representations
    root='./data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
)

train_loader = DataLoader(train_dataset_vicreg, batch_size=128, shuffle=True) # 128 to have at least 10 images of each class in a batch
train_loader_labels = DataLoader(train_dataset_labels, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VICRegModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# TRAINING
def train_step(images_views):
    x1, x2 = images_views[0].to(device), images_views[1].to(device)
    
    optimizer.zero_grad()
    z1 = model(x1)
    z2 = model(x2)
    
    loss = vicreg_loss(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()

print(f"Start VICReg Training in {device}...")
model.train()

for epoch in range(1, 21): 
    total_loss = 0
    for batch_idx, (images_views, _) in enumerate(train_loader):
        # images_views contains [view_1, view_2] 
        loss_val = train_step(images_views)
        total_loss += loss_val
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss_val:.4f}")
    
    print(f"==> Epoch {epoch} terminée. Loss moyenne: {total_loss/len(train_loader):.4f}")

print("Training completed.")


print("Evaluation")
evaluate(model, train_loader_labels, test_loader, device)