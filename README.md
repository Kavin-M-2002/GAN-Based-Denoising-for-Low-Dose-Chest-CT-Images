# GAN-Based-Denoising-for-Low-Dose-Chest-CT-Images
🔍 Project Title

GAN-Based Architecture for Enhancing Low-Dose CT Imaging Quality


📦 Installation & Setup

# Clone the repo or copy project files
mkdir "INTERNSHIP PROJECT"

cd "INTERNSHIP PROJECT"

# Create virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

✏️ requirements.txt (include this file)

torch
torchvision
numpy
matplotlib
Pillow
scikit-image

🚀 Execution Steps

1. Train the GAN model

python -m src.train

Trains for 100 epochs

Saves generator model to models/generator.pth

Saves loss curve to outputs/loss_curve.png

2. Denoise the test dataset

python -m src.test

Saves denoised images to outputs/predicted_images/

3. Evaluate PSNR and SSIM

python -m src.evaluate

Console output of average PSNR and SSIM

4. Visualize Random Sample Results

python visualization_results.py

Outputs comparison images to outputs/visual_results/

📊 Results

Metric

Value (Sample Output)

PSNR

~16.64 dB

SSIM

~0.6537

📌 Notes

Input images were artificially noised with Gaussian noise to simulate LDCT.

Dataset source: Kaggle Chest CT-Scan Images
