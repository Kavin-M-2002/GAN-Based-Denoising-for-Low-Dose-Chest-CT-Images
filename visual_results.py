import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from src.model import UNetGenerator
from src.dataset import CTScanDenoiseDataset

# -------------------- Configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/generator.pth"
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_BATCHES = 3  # Number of batches to visualize
SAVE_DIR = "outputs/visual_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- Load Model --------------------
gen = UNetGenerator().to(DEVICE)
gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
gen.eval()

# -------------------- Load Dataset --------------------
dataset = CTScanDenoiseDataset(root_dir="Data", image_size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------- Helper Function --------------------
def imshow_tensor(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img.squeeze()

# -------------------- Generate Visualizations --------------------
for batch_idx, (noisy, clean) in enumerate(loader):
    if batch_idx >= NUM_BATCHES:
        break

    noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
    with torch.no_grad():
        denoised = gen(noisy)

    fig, axes = plt.subplots(BATCH_SIZE, 3, figsize=(10, BATCH_SIZE * 3))
    column_titles = ["Noisy Input", "Denoised Output", "Clean Ground Truth"]

    for ax, col in zip(axes[0], column_titles):
        ax.set_title(col, fontsize=14)

    for i in range(BATCH_SIZE):
        imgs = [noisy[i], denoised[i], clean[i]]
        psnr_val = psnr(imshow_tensor(clean[i]), imshow_tensor(denoised[i]), data_range=1.0)
        ssim_val = ssim(imshow_tensor(clean[i]), imshow_tensor(denoised[i]), data_range=1.0)

        for j in range(3):
            axes[i][j].imshow(imshow_tensor(imgs[j]), cmap="gray")
            axes[i][j].axis("off")
            if j == 1:
                axes[i][j].set_xlabel(f"PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.2f}", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"result_{batch_idx + 1}.png")
    plt.savefig(save_path)
    plt.close()

print(f"✅ Visualizations saved in '{SAVE_DIR}'")
