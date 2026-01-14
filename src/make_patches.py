import os
import cv2
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def crop_patch(img, cx, cy, patch_size):
    h, w = img.shape[:2]
    r = patch_size // 2
    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r
    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        return None
    return img[y1:y2, x1:x2].copy()

def sample_points(mask, label_value, n):
    ys, xs = np.where(mask == label_value)
    if len(xs) == 0:
        return []
    idx = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    return list(zip(xs[idx], ys[idx]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="data/raw/images")
    ap.add_argument("--masks_dir", required=True, help="data/raw/masks")
    ap.add_argument("--out_dir", required=True, help="data/patches")
    ap.add_argument("--patch_size", type=int, default=64)
    ap.add_argument("--per_image_skin", type=int, default=200)
    ap.add_argument("--per_image_nonskin", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="80,10,10", help="train,val,test percentages")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    split = [int(x) for x in args.split.split(",")]
    assert sum(split) == 100, "split must sum to 100"

    # Prepare output folders
    for sp in ["train", "val", "test"]:
        ensure_dir(os.path.join(args.out_dir, sp, "skin"))
        ensure_dir(os.path.join(args.out_dir, sp, "nonskin"))

    # List files
    img_files = sorted([f for f in os.listdir(args.images_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not img_files:
        raise RuntimeError("No images found in images_dir")

    # Shuffle and split
    random.shuffle(img_files)
    n = len(img_files)
    n_train = int(n * split[0] / 100)
    n_val = int(n * split[1] / 100)
    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train + n_val]
    test_files = img_files[n_train + n_val:]

    def process(files, split_name):
        saved = {"skin": 0, "nonskin": 0}
        for f in tqdm(files, desc=f"Creating patches [{split_name}]"):
            img_path = os.path.join(args.images_dir, f)
            base, _ = os.path.splitext(f)

            # mask must have same base name (allow png/jpg)
            mask_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(args.masks_dir, base + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                # skip if no mask
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Normalize mask to {0,255} then map to {0,1}
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            skin_points = sample_points(mask_bin, 255, args.per_image_skin)
            nonskin_points = sample_points(mask_bin, 0, args.per_image_nonskin)

            # Save patches
            for (x, y) in skin_points:
                patch = crop_patch(img, x, y, args.patch_size)
                if patch is None:
                    continue
                out = os.path.join(args.out_dir, split_name, "skin",
                                   f"{base}_s_{x}_{y}.png")
                cv2.imwrite(out, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                saved["skin"] += 1

            for (x, y) in nonskin_points:
                patch = crop_patch(img, x, y, args.patch_size)
                if patch is None:
                    continue
                out = os.path.join(args.out_dir, split_name, "nonskin",
                                   f"{base}_n_{x}_{y}.png")
                cv2.imwrite(out, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                saved["nonskin"] += 1

        return saved

    stats = {
        "train": process(train_files, "train"),
        "val": process(val_files, "val"),
        "test": process(test_files, "test"),
        "patch_size": args.patch_size,
        "per_image_skin": args.per_image_skin,
        "per_image_nonskin": args.per_image_nonskin,
        "split": {"train": split[0], "val": split[1], "test": split[2]}
    }

    ensure_dir("outputs")
    with open("outputs/patch_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Done. Patch stats saved to outputs/patch_stats.json")
    print(stats)

if __name__ == "__main__":
    main()
