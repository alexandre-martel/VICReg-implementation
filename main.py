import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import os
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt


from src.TwoViewTransformClass import TwoViewsTransform
from src.VICRegModelClass import VICRegModel
from src.VICRegLoss import vicreg_loss
from src.evaluate import evaluate

parser = argparse.ArgumentParser(description="VICReg Training and Evaluation")
parser.add_argument("mode", choices=["TRAIN", "RE_TRAIN", "EVALUATE", "INFERENCE"], help="Execution mode")
parser.add_argument("--epochs", type=int, default=20, help="epoch number for training")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--model_path", type=str, help="path to the .pth model (required for RE_TRAIN, EVAL, INF)")
parser.add_argument("--name", type=str, default="vicreg_model.pth", help="model save name (for TRAIN and RE_TRAIN modes)")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_loader = DataLoader(train_dataset_vicreg, batch_size=args.batch_size, shuffle=True) # 128 to have at least 10 images of each class in a batch
train_loader_labels = DataLoader(train_dataset_labels, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VICRegModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if args.mode in ["RE_TRAIN", "EVALUATE", "INFERENCE"]:
    if not args.model_path:
        raise ValueError(f"The mode {args.mode} requires --model_path")
    
    # Load the weights into the backbone
    state_dict = torch.load(args.model_path, map_location=device)
    model.backbone.load_state_dict(state_dict)
    print(f"Model loaded from {args.model_path}")

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

#  MODE TRAIN, RE_TRAIN
if args.mode in ["TRAIN", "RE_TRAIN"]:
    print(f"Start VICReg Training ({args.mode}) for {args.epochs} epochs...")
    loss_val = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for batch_idx, (images_views, _) in enumerate(train_loader):
            loss_val = train_step(images_views)
            total_loss += loss_val
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss_val:.4f}")
        loss_val.append(loss_val)
        print(f"==> Epoch {epoch} finished. mean loss: {total_loss/len(train_loader):.4f}")


    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", args.name)
    torch.save(model.backbone.state_dict(), save_path)
    print(f"Model saved in {save_path}")

    # Display loss curve
    plt.plot(range(1, len(loss_val) + 1), loss_val)
    plt.xlabel("Epochs")
    plt.ylabel("VICReg Loss")
    plt.title("VICReg Training Loss Curve")
    plt.show()


# MODE EVALUATE
if args.mode == "EVALUATE":
    evaluate(model, train_loader_labels, test_loader, device)

# MODE INFERENCE
elif args.mode == "INFERENCE":
    model.eval()

    img, _ = next(iter(test_loader))
    with torch.no_grad():
        feature_vector = model.backbone(img[0:1].to(device))
    print(f"Inference success : {feature_vector.shape}")

