import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Dossiers
IMG_DIR = Path("data/processed/128/test/images")
MASK_ROOT = Path("data/processed/masks")
OUT_DIR = Path("results/baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ratios = [10, 20, 40]
methods = {
    "Telea": cv2.INPAINT_TELEA,
    "NS": cv2.INPAINT_NS
}

results = []

# On récupère la liste des images de test
img_paths = list(IMG_DIR.glob("*.png"))
print(f"Évaluation de la baseline sur {len(img_paths)} images de test...")

for ratio in ratios:
    mask_dir = MASK_ROOT / f"test_square_{ratio}"
    
    for img_path in tqdm(img_paths, desc=f"Masques {ratio}%"):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        
        if not mask_path.exists():
            continue
            
        # Charger l'image (BGR) et le masque (Grayscale)
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Créer une version trouée de l'image
        img_masked = img.copy()
        img_masked[mask > 0] = 0 
        
        for method_name, method_flag in methods.items():
            # Appliquer l'inpainting
            inpainted = cv2.inpaint(img_masked, mask, 3, method_flag)
            
            # Calculer les métriques (sur toute l'image)
            # On compare l'image reconstruite avec l'image originale
            score_psnr = psnr(img, inpainted, data_range=255)
            # Pour SSIM, on doit spécifier channel_axis pour les images couleur
            score_ssim = ssim(img, inpainted, data_range=255, channel_axis=2)
            
            results.append({
                "ratio": f"{ratio}%",
                "method": method_name,
                "image": stem,
                "psnr": score_psnr,
                "ssim": score_ssim
            })
            
            # Sauvegarder quelques exemples visuels (juste pour le rapport)
            if stem in [img_paths[0].stem, img_paths[1].stem, img_paths[2].stem]:
                out_img_path = OUT_DIR / f"{stem}_{ratio}_{method_name}.png"
                cv2.imwrite(str(out_img_path), inpainted)

# Sauvegarder les métriques dans un CSV
df = pd.DataFrame(results)
df.to_csv("results/baseline_metrics.csv", index=False)

# Afficher un résumé (moyenne par méthode et par ratio)
summary = df.groupby(["ratio", "method"])[["psnr", "ssim"]].mean().reset_index()
print("\n--- RÉSULTATS MOYENS ---")
print(summary.to_string(index=False))
print("\nLes résultats détaillés ont été sauvegardés dans results/baseline_metrics.csv")
print("Des exemples visuels sont dans le dossier results/baseline/")
