import torch
import torch.nn as nn

def evaluate(model, train_loader_labels, test_loader, device):
    print("\n--- Evaluation ---")
    model.backbone.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False # freeze ResNet
    
    # Simple classifier : 512 features in input -> 10 classes in output
    classifier = nn.Linear(512, 10).to(device)
    optim_eval = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for e in range(5): # 5 epochs are enough for the classifier
        # NOTE: We train only the classifier here, the backbone is frozen. The classifier is small because the 
        # VICReg model should have learned good representations, so it shouldn't need much capacity to classify.
        for imgs, labels in train_loader_labels:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = model.backbone(imgs)
            logits = classifier(feat)
            loss = criterion(logits, labels)
            
            optim_eval.zero_grad()
            loss.backward()
            optim_eval.step()
        print(f"Evaluation epoch {e+1}/5 completed.")

    # Final accuracy
    correct, total = 0, 0
    classifier.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = model.backbone(imgs)
            preds = classifier(feat).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    print(f"\nFinal accuracy on CIFAR-10: {100 * correct / total:.2f}%")

