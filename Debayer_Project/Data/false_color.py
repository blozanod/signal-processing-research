import numpy as np
import os
from PIL import Image

# Current Filepath
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DIV2K_DIR = os.path.join(CURRENT_PATH, "DIV2K_Images")
DIV2K_VALID_BAYER = os.path.join(DIV2K_DIR, "DIV2K_Valid_HR")

MODEL_PATH = "../model.pth"

# Image
image_path = os.path.join(DIV2K_VALID_BAYER, "0867.png")
input_image = Image.open(image_path).convert("RGB")

# Input image
rgb = np.array(input_image)  # shape (H, W, 3)

H, W, _ = rgb.shape

# False Color Image
false_color = np.zeros((H, W, 3), dtype=np.uint8)

# Bayer pattern RGGB
false_color[0::2, 0::2, 0] = rgb[0::2, 0::2, 0] # R at (0,0)
false_color[0::2, 1::2, 1] = rgb[0::2, 1::2, 1] # G at (0,1)
false_color[1::2, 0::2, 1] = rgb[1::2, 0::2, 1] # G at (1,0)
false_color[1::2, 1::2, 2] = rgb[1::2, 1::2, 2] # B at (1,1)

# Center crop
crop_size = 64

i = (H - crop_size) // 2
j = (W - crop_size) // 2
cropped = false_color[i:i+crop_size, j:j+crop_size]

# Input crop
input_crop = input_image.crop((j, i, j + crop_size, i + crop_size))

# Save
input_crop.save("input_crop.png")
Image.fromarray(cropped).save("false_color.png")