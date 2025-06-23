import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import CTScanDenoiseDataset
from src.model import UNetGenerator, PatchDiscriminator
from src.perceptual_loss import VGGFeatureExtractor
import os
import matplotlib.pyplot as plt

# -------------------- Config --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.0002
IMG_SIZE = 256
MODEL_PATH = "models/generator.pth"
PERCEPTUAL_WEIGHT = 0.01  

# -------------------- Setup --------------------
gen = UNetGenerator().to(DEVICE)
disc = PatchDiscriminator().to(DEVICE)
vgg_feat = VGGFeatureExtractor().to(DEVICE).eval()

opt_G = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE)
opt_D = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE)

bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

# -------------------- Dataset --------------------
data = CTScanDenoiseDataset(root_dir="Data", image_size=IMG_SIZE)
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# -------------------- Track losses --------------------
G_losses = []
D_losses = []

# -------------------- Training Loop --------------------
for epoch in range(EPOCHS):
    for batch_idx, (noisy_img, clean_img) in enumerate(loader):
        noisy_img, clean_img = noisy_img.to(DEVICE), clean_img.to(DEVICE)

        # Labels
        disc_out = disc(clean_img)
        valid = torch.ones_like(disc_out, device=DEVICE)
        fake = torch.zeros_like(disc_out, device=DEVICE)

        # Generator
        opt_G.zero_grad()
        generated_img = gen(noisy_img)
        pred_fake = disc(generated_img)

        # Repeat grayscale to 3 channels for VGG
        gen_vgg = vgg_feat(generated_img.repeat(1, 3, 1, 1))
        cln_vgg = vgg_feat(clean_img.repeat(1, 3, 1, 1))

        # Losses
        loss_perceptual = mse_loss(gen_vgg, cln_vgg)
        loss_G = (
            bce_loss(pred_fake, valid) +
            100 * l1_loss(generated_img, clean_img) +
            PERCEPTUAL_WEIGHT * loss_perceptual
        )
        loss_G.backward()
        opt_G.step()

        # Discriminator
        opt_D.zero_grad()
        loss_real = bce_loss(disc(clean_img), valid)
        loss_fake = bce_loss(disc(generated_img.detach()), fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        opt_D.step()

    # Track loss per epoch
    G_losses.append(loss_G.item())
    D_losses.append(loss_D.item())

    print(f"[Epoch {epoch+1}/{EPOCHS}]  Loss_G: {loss_G.item():.4f} | Loss_D: {loss_D.item():.4f}")

# -------------------- Save Model --------------------
os.makedirs("models", exist_ok=True)
torch.save(gen.state_dict(), MODEL_PATH)
print(f"✅ Generator saved to '{MODEL_PATH}'")

# -------------------- Visualize Loss Curves --------------------
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs (with Perceptual Loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/loss_curve.png")
print("📈 Loss curve saved to 'outputs/loss_curve.png'")
