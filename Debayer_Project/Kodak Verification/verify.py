import os
import sys
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2 # Used for the "Baseline" comparison

# 1. Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the parent directory (where 'Model' folder is located)
parent_dir = os.path.dirname(current_dir)
# 3. Add parent directory to sys.path
sys.path.append(parent_dir)

from Model.unet import UNet
import torch
from torchvision import transforms
import requests

MODEL_PATH = "../model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KODAK_FOLDER = "Kodak_Images"
DOWNLOAD_URL_BASE = "http://r0k.us/graphics/kodak/kodak/"

def download_kodak(destination_folder):
    """
    Checks if Kodak images exist. If not, downloads them.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Directory '{destination_folder}' created.")
    
    # Check if folder is empty or missing files (should be 24 images)
    existing_files = [f for f in os.listdir(destination_folder) if f.endswith('.png')]
    
    if len(existing_files) == 24:
        print(f"Kodak dataset already present in '{destination_folder}'. Skipping download.")
        return

    print(f"Downloading Kodak dataset to '{destination_folder}'...")
    
    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        file_path = os.path.join(destination_folder, filename)
        
        if not os.path.exists(file_path):
            try:
                url = DOWNLOAD_URL_BASE + filename
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f" - Downloaded {filename}")
            except Exception as e:
                print(f" ! Failed to download {filename}: {e}")

    print("Download verification complete.")

def mosaic_image(image, pattern='RGGB'):
    """
    Applies a Bayer Mosaic to a ground truth image.
    Returns: A single-channel numpy array (H, W)
    """
    H, W, C = image.shape
    mosaic = np.zeros((H, W), dtype=image.dtype)

    # RGGB Pattern masks
    # Red: (0,0), Green: (0,1) & (1,0), Blue: (1,1)
    if pattern == 'RGGB':
        mosaic[0::2, 0::2] = image[0::2, 0::2, 0] # Red
        mosaic[0::2, 1::2] = image[0::2, 1::2, 1] # Green 1
        mosaic[1::2, 0::2] = image[1::2, 0::2, 1] # Green 2
        mosaic[1::2, 1::2] = image[1::2, 1::2, 2] # Blue
    # You can add 'GRBG' or other patterns if your model uses them
    
    return mosaic

def transform_image(mosaic):
    transform = transforms.ToTensor()

    bayer_array = np.array(mosaic)
    h, w = bayer_array.shape
    h = (h // 2) * 2
    w = (w // 2) * 2
    bayer_array = bayer_array[:h, :w]

    channel_r  = bayer_array[0::2, 0::2]  # Red
    channel_g1 = bayer_array[0::2, 1::2]  # Green 1
    channel_g2 = bayer_array[1::2, 0::2]  # Green 2
    channel_b  = bayer_array[1::2, 1::2]  # Blue

    input_4channel = np.stack([channel_r, channel_g1, channel_g2, channel_b], axis=-1)

    input_tensor = transform(input_4channel).unsqueeze(0).to(DEVICE)

    return input_tensor, h, w

def run_validation(kodak_folder_path, model):
    psnr_scores = []
    ssim_scores = []
    baseline_psnr_scores = []
    
    files = sorted([f for f in os.listdir(kodak_folder_path) if f.endswith('.png')])
    
    print(f"Found {len(files)} images. Starting validation...")

    for filename in files:
        # 1. Load Ground Truth (y)
        gt_path = os.path.join(kodak_folder_path, filename)
        gt_img = np.array(Image.open(gt_path).convert('RGB'))
        
        # 2. Create Input (x) by Mosaicing
        input_mosaic = mosaic_image(gt_img, pattern='RGGB')
        
        # --- MODEL PREDICTION ---
        input_tensor, h_crop, w_crop = transform_image(input_mosaic)
        
        # Prediction should be (H, W, 3)
        with torch.no_grad():
            prediction = model(input_tensor)

        prediction = prediction.squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Convert prediction back to uint8 (0-255) for metric calc
        prediction_uint8 = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)

        gt_img_cropped = gt_img[:h_crop, :w_crop, :]
        
        # --- BASELINE (OPENCV) ---
        # OpenCV expects uint8 input. We use edge-aware demosaicing (VNG) or Bilinear
        input_mosaic_cropped = input_mosaic[:h_crop, :w_crop]
        baseline_prediction = cv2.cvtColor(input_mosaic_cropped, cv2.COLOR_BayerBG2RGB) # Note: BG matches RGGB reversed
        
        # 3. Calculate Metrics
        score_p = psnr(gt_img_cropped, prediction_uint8, data_range=255)
        score_s = ssim(gt_img_cropped, prediction_uint8, data_range=255, channel_axis=-1)

        baseline_p = psnr(gt_img_cropped, baseline_prediction, data_range=255)

        psnr_scores.append(score_p)
        ssim_scores.append(score_s)
        baseline_psnr_scores.append(baseline_p)

    # 4. Final Results for Resume
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_baseline = np.mean(baseline_psnr_scores)

    print("-" * 30)
    print(f"Validation Complete on {len(files)} Images")
    print(f"Baseline (OpenCV) PSNR: {avg_baseline:.2f} dB")
    print(f"Your Model PSNR:        {avg_psnr:.2f} dB")
    print(f"Your Model SSIM:        {avg_ssim:.4f}")
    print("-" * 30)
    print(f"Improvement: +{avg_psnr - avg_baseline:.2f} dB")


# Load Model and Run Verification
print(f"Using device: {DEVICE}")
print(f"Loading model from {MODEL_PATH}...")
model = UNet().to(DEVICE)

state_dict = torch.load(MODEL_PATH)
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

model.load_state_dict(new_state_dict)
model.eval()

download_kodak(KODAK_FOLDER)
run_validation(KODAK_FOLDER, model)