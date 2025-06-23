# GAN-Based-Denoising-for-Low-Dose-Chest-CT-Images
🔍 Project Title

GAN-Based Architecture for Enhancing Low-Dose CT Imaging Quality

📁 Project Structure

INTERNSHIP PROJECT/
├── venv/                     # Virtual environment
├── Data/                     # Dataset: train/valid/test (4 classes)
├── models/                   # Trained model weights (generator.pth)
├── outputs/
│   ├── loss_curve.png        # Loss trend plot
│   ├── predicted_images/     # Denoised results from test set
│   └── visual_results/       # Visualization with PSNR/SSIM
├── src/
│   ├── dataset.py            # Custom PyTorch dataset loader
│   ├── model.py              # Generator & Discriminator architecture
│   ├── train.py              # Training loop
│   ├── test.py               # Image denoising inference
│   └── evaluate.py           # PSNR/SSIM calculation
├── visualization_results.py # Visualization of results
└── README.md                 # Project documentation

📦 Installation & Setup

# Clone the repo or copy project files
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
