import streamlit as st
st.set_page_config(page_title="CT Denoising GAN", layout="wide")
import torch
from torchvision import transforms
from PIL import Image
from src.model import UNetGenerator
from src.utils import load_image, tensor_to_image, add_gaussian_noise, compute_metrics

# -------------------- Config --------------------
MODEL_PATH = "models/generator.pth"
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = UNetGenerator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

gen = load_model()

# -------------------- UI Layout --------------------
st.title("🩻 CT Scan Denoising using GAN")
st.markdown("Upload a noisy CT scan image and view the denoised result using a GAN model.")

st.sidebar.header("Project Info")
st.sidebar.markdown("""
- **Model:** GAN with U-Net Generator  
- **Dataset:** Lung Cancer CT Scans  
- **Trained on:** 100 epochs  
- **Metrics:** PSNR, SSIM  
""")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("Upload CT Scan Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.subheader("📥 Uploaded Image")
    col1, col2, col3 = st.columns(3)

    # Load and preprocess image
    clean_tensor = load_image(uploaded_file, IMG_SIZE).to(DEVICE)
    noisy_tensor = add_gaussian_noise(clean_tensor)

    with torch.no_grad():
        denoised_tensor = gen(noisy_tensor)

    # Convert tensors to images
    noisy_img = tensor_to_image(noisy_tensor)
    denoised_img = tensor_to_image(denoised_tensor)
    clean_img = tensor_to_image(clean_tensor)

    # Metrics
    psnr, ssim = compute_metrics(denoised_tensor, clean_tensor)

    # Display images
    with col1:
        st.image(noisy_img, caption="Noisy Input", use_column_width=True)
    with col2:
        st.image(denoised_img, caption="Denoised Output", use_column_width=True)
    with col3:
        st.image(clean_img, caption="Clean Ground Truth", use_column_width=True)

    # Show metrics
    st.markdown(f"📊 **PSNR:** {psnr:.2f} dB &nbsp;&nbsp;&nbsp; **SSIM:** {ssim:.4f}")
