import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.utils
import numpy as np

from Model.unet import UNet

# --- Configuration ---
# MODEL_PATH = "model_checkpoint.pth"
MODEL_PATH = "model.pth"

VERSION = "v3"
# Images to be generated
IMAGE_PATHS = ["0801","0802","0808","0844","0852","0873","0898"]

for image in IMAGE_PATHS:
    # Image path
    BAYER_IMAGE_PATH = "Data/DIV2K_Images/DIV2K_Valid_Bayer/" + image + ".png"
    OUTPUT_IMAGE_PATH = image + VERSION + ".png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {MODEL_PATH}...")
    model = UNet().to(DEVICE)

    state_dict = torch.load(MODEL_PATH)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(new_state_dict)
    # checkpoint = torch.load(MODEL_PATH)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loading image {BAYER_IMAGE_PATH}...")

    transform = transforms.ToTensor()

    pil_image = Image.open(BAYER_IMAGE_PATH).convert('L')
    bayer_array = np.array(pil_image) # This is now a 2D numpy array
    h, w = bayer_array.shape            # This will now work
    channel_r = np.zeros_like(bayer_array)
    channel_g1 = np.zeros_like(bayer_array)
    channel_g2 = np.zeros_like(bayer_array)
    channel_b = np.zeros_like(bayer_array)

    channel_r[0::2, 0::2] = bayer_array[0::2, 0::2]  # Red
    channel_g1[0::2, 1::2] = bayer_array[0::2, 1::2] # Green 1
    channel_g2[1::2, 0::2] = bayer_array[1::2, 0::2] # Green 2
    channel_b[1::2, 1::2] = bayer_array[1::2, 1::2]  # Blue

    input_4channel = np.stack([channel_r, channel_g1, channel_g2, channel_b], axis=-1)

    input_tensor = transform(input_4channel).unsqueeze(0).to(DEVICE)

    print(f"Running inference on image with shape: {input_tensor.shape}")
    # Run inference
    with torch.no_grad():
        # The model's internal padding/cropping logic will handle the non-256x256 size
        output_tensor = model(input_tensor)

    print(f"Saving output to {OUTPUT_IMAGE_PATH}...")
    # Save the output
    torchvision.utils.save_image(output_tensor[0], OUTPUT_IMAGE_PATH)

    print("Done!")