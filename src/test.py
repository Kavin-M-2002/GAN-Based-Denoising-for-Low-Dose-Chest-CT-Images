import os
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from src.dataset import CTScanDenoiseDataset
from src.model import UNetGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNetGenerator().to(device)
model.load_state_dict(torch.load("models/generator.pth", map_location=device))
model.eval()

# Load test data
test_data = CTScanDenoiseDataset(root_dir="Data", subset="test", image_size=256)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

os.makedirs("outputs/denoised", exist_ok=True)

# Convert tensor to PIL
to_pil = T.ToPILImage()

for i, (noisy, clean) in enumerate(test_loader):
    noisy = noisy.to(device)
    with torch.no_grad():
        denoised = model(noisy)

    # Convert to PIL
    noisy_img = to_pil(noisy[0].cpu())
    denoised_img = to_pil(denoised[0].cpu())
    clean_img = to_pil(clean[0].cpu())

    # Create a blank canvas
    width, height = noisy_img.width, noisy_img.height
    combined = Image.new("L", (width * 3, height + 30), color=255)

    # Paste images
    combined.paste(noisy_img, (0, 30))
    combined.paste(denoised_img, (width, 30))
    combined.paste(clean_img, (width * 2, 30))

    # Draw labels
    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default()
    draw.text((width // 2 - 30, 5), "Noisy Input", fill=0, font=font)
    draw.text((width + width // 2 - 40, 5), "Denoised Output", fill=0, font=font)
    draw.text((2 * width + width // 2 - 50, 5), "Clean Ground Truth", fill=0, font=font)

    # Save image
    combined.save(f"outputs/denoised/sample_{i}.png")

print("✅ Labeled predictions saved in outputs/denoised/")
