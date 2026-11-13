import os, cv2, json, random, numpy as np, kagglehub
from pathlib import Path
from tqdm import tqdm

print("Téléchargement du dataset depuis Kaggle (environ 37 Mo)...")
path = kagglehub.dataset_download("anassiguedmi/moroccan-zellige")
print("Dataset téléchargé dans :", path)

# 1. Inventaire des images
exts = ('.jpg', '.jpeg', '.png', '.webp')
image_paths = [os.path.join(r, f) for r, d, files in os.walk(path) for f in files if f.lower().endswith(exts)]
random.seed(42)
random.shuffle(image_paths)
print(f"Total: {len(image_paths)} images trouvées.")

# 2. Split 70/15/15
n = len(image_paths)
splits = {
    "train": image_paths[:int(0.70 * n)],
    "val": image_paths[int(0.70 * n):int(0.85 * n)],
    "test": image_paths[int(0.85 * n):]
}
print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

# 3. Redimensionnement 128x128
SIZE = 128
OUT = Path("data/processed/128")
for split, plist in splits.items():
    out_dir = OUT / split / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(plist, desc=f"Redimensionnement {split}"):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / (Path(p).stem + ".png")), img)

# 4. Génération des masques (zones trouées)
def square_mask(size, ratio, rng):
    area = int(ratio * size * size)
    side = int(np.sqrt(area))
    m = np.zeros((size, size), np.uint8)
    x, y = rng.integers(0, size - side + 1, 2)
    m[y:y+side, x:x+side] = 1
    return m

MASK_ROOT = Path("data/processed/masks")
for split, plist in splits.items():
    img_dir = OUT / split / "images"
    for r in [0.10, 0.20, 0.40]:
        out = MASK_ROOT / f"{split}_square_{int(r*100)}"
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42 + int(r*100))
        for p in tqdm(plist, desc=f"Masques {split} {int(r*100)}%"):
            stem = Path(p).stem
            if (img_dir / f"{stem}.png").exists():
                cv2.imwrite(str(out / f"{stem}.png"), square_mask(SIZE, r, rng) * 255)

# 5. Sauvegarde JSON et CSV
with open("data/splits.json", "w") as f:
    json.dump({k: [Path(p).stem for p in v] for k, v in splits.items()}, f)

import pandas as pd
pd.DataFrame({"path": image_paths, "source": "kaggle:moroccan-zellige", "license": "A verifier"}).to_csv("data/licenses.csv", index=False)
print("Terminé avec succès ! Les données sont prêtes.")

