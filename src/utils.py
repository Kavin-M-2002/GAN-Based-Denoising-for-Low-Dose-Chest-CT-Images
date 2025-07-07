# utils.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import lpips
from skimage.metrics import structural_similarity as ssim
import math

def load_image(uploaded_file, img_size):
    image = Image.open(uploaded_file).convert("L")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def tensor_to_image(tensor):
    image = tensor.squeeze(0).detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image[0])

def add_gaussian_noise(tensor, mean=0.0, std=0.1):
    noisy = tensor + torch.randn_like(tensor) * std
    return torch.clamp(noisy, 0., 1.)

def compute_metrics(output, target):
    output_np = output.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()

    psnr_val = compute_psnr(output_np, target_np)
    ssim_val = ssim(target_np, output_np, data_range=1.0)
    return psnr_val, ssim_val

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))
