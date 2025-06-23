import torch, numpy as np
from PIL import Image
import math
from skimage.metrics import structural_similarity as ssim

def tensor_to_image(t):
    arr = (t.squeeze().cpu().numpy()*255).astype(np.uint8)
    return Image.fromarray(arr)

def compute_psnr_ssim(gt, pred):
    gt, pred = gt.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy()
    mse = np.mean((gt-pred)**2)
    psnr = 20 * math.log10(1.0/math.sqrt(mse+1e-8))
    s = ssim(gt, pred, data_range=1.0)
    return psnr, s

def save_comparison(no, out, cl, path):
    img_no, img_out, img_cl = tensor_to_image(no), tensor_to_image(out), tensor_to_image(cl)
    w, h = img_no.width, img_no.height
    comp = Image.new('L', (w*3, h))
    comp.paste(img_no, (0,0)); comp.paste(img_out,(w,0)); comp.paste(img_cl,(2*w,0))
    comp.save(path)
