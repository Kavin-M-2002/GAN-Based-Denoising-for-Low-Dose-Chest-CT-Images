import os
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms

def add_gaussian_noise(np_img, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    noisy = np_img + np.random.normal(mean, sigma, np_img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)

class CTScanDenoiseDataset(Dataset):
    def __init__(self, root_dir, image_size=256, subset='train'):
       
        self.pairs = []
        assert subset in ["train", "valid", "test"], "subset must be 'train', 'valid', or 'test'"
        split_dir = os.path.join(root_dir, subset)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            image_paths = glob(os.path.join(cls_dir, "*.png")) + glob(os.path.join(cls_dir, "*.jpg"))
            for img_path in image_paths:
                clean_img = Image.open(img_path).convert("L")  # Grayscale
                noisy_img = Image.fromarray(
                    add_gaussian_noise(np.array(clean_img))
                )
                self.pairs.append((noisy_img, clean_img))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        return self.transform(noisy), self.transform(clean)
