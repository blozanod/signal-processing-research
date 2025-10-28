import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.utils

from Model import unet as un

# --- Configuration ---
MODEL_PATH = "model_checkpoint.pth"
# Image path
BAYER_IMAGE_PATH = "Data/DIV2K_Images/DIV2K_Valid_Bayer/0852.png" 
OUTPUT_IMAGE_PATH = "output.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
model = un.UNet().to(DEVICE)
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loading image {BAYER_IMAGE_PATH}...")

transform = transforms.ToTensor()

pil_image = Image.open(BAYER_IMAGE_PATH).convert('L')
input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

print(f"Running inference on image with shape: {input_tensor.shape}")
# Run inference
with torch.no_grad():
    # The model's internal padding/cropping logic will handle the non-256x256 size
    output_tensor = model(input_tensor)

print(f"Saving output to {OUTPUT_IMAGE_PATH}...")
# Save the output
torchvision.utils.save_image(output_tensor[0], OUTPUT_IMAGE_PATH)

print("Done!")