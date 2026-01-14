import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

"""
Module: predict.py
Description:
    Command-line inference script. Takes a single image path and a model checkpoint,
    and outputs the predicted class (Skin/Non-Skin) and probabilities.
"""

def load_model(ckpt_path, device):
    """
    Loads the trained model for inference.
    
    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (str): Computation device.
        
    Returns:
        tuple: (model, idx_to_class, img_size)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    class_map = ckpt["class_map"]
    img_size = ckpt["img_size"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    idx_to_class = {v: k for k, v in class_map.items()}
    return model, idx_to_class, img_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/model.pt")
    ap.add_argument("--image", required=True, help="path to image")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, idx_to_class, img_size = load_model(args.ckpt, device)

    tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(prob.argmax())

    print(f"Prediction: {idx_to_class[pred]}")
    print(f"Probabilities: {prob}")

if __name__ == "__main__":
    main()
