import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate(model, train_loader_labels, test_loader, device):
    print("\n--- Phase d'évaluation : Entraînement du classifieur linéaire ---")
    model.backbone.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False # On gèle le ResNet
    
    # Simple couche linéaire : 512 caractéristiques en entrée -> 10 classes en sortie
    classifier = nn.Linear(512, 10).to(device)
    optim_eval = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for e in range(5): # 5 époques suffisent pour le classifieur
        for imgs, labels in train_loader_labels:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = model.backbone(imgs)
            logits = classifier(feat)
            loss = criterion(logits, labels)
            
            optim_eval.zero_grad()
            loss.backward()
            optim_eval.step()
        print(f"Époque d'évaluation {e+1}/5 complétée.")

    # Calcul de la précision finale
    correct, total = 0, 0
    classifier.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = model.backbone(imgs)
            preds = classifier(feat).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    print(f"\nPrécision finale sur CIFAR-10 : {100 * correct / total:.2f}%")

# Lancer l'évaluation
evaluate(model, train_loader_labels, test_loader)