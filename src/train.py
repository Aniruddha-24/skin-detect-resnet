import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/patches")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=val_tfms)

    # class_to_idx should map: {"nonskin":0, "skin":1} or vice versa depending on folder order
    class_map = train_ds.class_to_idx

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model: ResNet18 transfer learning
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "class_map": class_map}

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses, tr_accs = [], []

        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_losses.append(loss.item())
            tr_accs.append(accuracy_from_logits(logits, y))

        model.eval()
        va_losses, va_accs = [], []
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_losses.append(loss.item())
                va_accs.append(accuracy_from_logits(logits, y))

        train_loss = float(np.mean(tr_losses))
        train_acc  = float(np.mean(tr_accs))
        val_loss   = float(np.mean(va_losses))
        val_acc    = float(np.mean(va_accs))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Save best weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_map": class_map,
                "img_size": args.img_size
            }, os.path.join(args.out_dir, "model.pt"))
            print(f"Saved new best model to {os.path.join(args.out_dir,'model.pt')}")

    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("Training done.")
    print("Class map:", class_map)

if __name__ == "__main__":
    main()
