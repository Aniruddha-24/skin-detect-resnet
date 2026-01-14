import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_map = ckpt["class_map"]
    img_size = ckpt["img_size"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_map, img_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/patches")
    ap.add_argument("--ckpt", default="outputs/model.pt")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--num_samples_plot", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, class_map, img_size = load_model(args.ckpt, device)
    idx_to_class = {v: k for k, v in class_map.items()}

    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=tfms)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred = [], []
    probs = []

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            pred = np.argmax(p, axis=1)

            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.tolist())
            probs.extend(p.tolist())

    # Metrics
    report = classification_report(y_true, y_pred, target_names=[idx_to_class[0], idx_to_class[1]], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(report["accuracy"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "idx_to_class": idx_to_class
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_class[0], idx_to_class[1]])
    disp.plot(values_format="d")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # Sample predictions grid
    # reload dataset without normalization for display
    display_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    display_ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=display_tfms)

    n = min(args.num_samples_plot, len(display_ds))
    idxs = np.random.choice(len(display_ds), size=n, replace=False)

    fig, axes = plt.subplots(int(np.ceil(n/4)), 4, figsize=(12, 3 * int(np.ceil(n/4))))
    axes = np.array(axes).reshape(-1)

    for i, ds_idx in enumerate(idxs):
        img, label = display_ds[ds_idx]

        # Run prediction with normalized tensor
        x = tfms(transforms.ToPILImage()(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            pred = int(np.argmax(p))

        axes[i].imshow(img.permute(1,2,0).numpy())
        axes[i].axis("off")
        axes[i].set_title(f"T:{idx_to_class[label]}  P:{idx_to_class[pred]}\n"
                          f"p_skin={p[class_map.get('skin',1)]:.2f}")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "sample_preds.png"), dpi=200)
    plt.close()

    print("Saved metrics to outputs/metrics.json")
    print("Saved confusion matrix to outputs/confusion_matrix.png")
    print("Saved sample predictions to outputs/sample_preds.png")

if __name__ == "__main__":
    main()
