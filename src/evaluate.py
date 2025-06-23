import os
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToPILImage
import numpy as np
from src.dataset import CTScanDenoiseDataset
from src.model import UNetGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator
gen = UNetGenerator().to(device)
gen.load_state_dict(torch.load("models/generator.pth", map_location=device))
gen.eval()

# Load test data
test_data = CTScanDenoiseDataset(root_dir="Data", subset="test", image_size=256)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

to_pil = ToPILImage()

psnr_scores, ssim_scores = [], []

for noisy, clean in test_loader:
    noisy = noisy.to(device)
    with torch.no_grad():
        denoised = gen(noisy)

    clean_img = to_pil(clean[0].cpu())
    denoised_img = to_pil(denoised[0].cpu())

    clean_np = np.array(clean_img)
    denoised_np = np.array(denoised_img)

    psnr_score = psnr(clean_np, denoised_np, data_range=255)
    ssim_score = ssim(clean_np, denoised_np, data_range=255)

    psnr_scores.append(psnr_score)
    ssim_scores.append(ssim_score)

# Print average
print(f"🔍 Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"🔍 Average SSIM: {np.mean(ssim_scores):.4f}")
