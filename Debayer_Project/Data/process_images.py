import numpy as np
import os
from PIL import Image

# Current Filepath
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# DIV 2K
DIV2K_DIR = os.path.join(CURRENT_PATH, "DIV2K_Images")
DIV2K_TRAIN_HR = os.path.join(DIV2K_DIR, "DIV2K_Train_HR")
DIV2K_VALID_HR = os.path.join(DIV2K_DIR, "DIV2K_Valid_HR")
DIV2K_TRAIN_BAYER = os.path.join(DIV2K_DIR, "DIV2K_Train_Bayer")
DIV2K_VALID_BAYER = os.path.join(DIV2K_DIR, "DIV2K_Valid_Bayer")

def bayer_img(img_pil, name, directory):
    if os.path.exists(os.path.join(directory,name)):
        print(f"Bayer image {name} previously processed, skipping...")
        return 0

    # Convert the PIL Image (RGB) to a NumPy array (RGB)
    img_array = np.array(img_pil) 
    h, w, channels = img_array.shape

    mossaic = np.zeros((h, w), dtype=img_array.dtype)

    # RGGB Mossaic pattern
    mossaic[0:h:2, 0:w:2] = img_array[0:h:2, 0:w:2, 0] # Top-Left = RED
    mossaic[0:h:2, 1:w:2] = img_array[0:h:2, 1:w:2, 1] # Top-Right = GREEN
    mossaic[1:h:2, 0:w:2] = img_array[1:h:2, 0:w:2, 1] # Bottom-Left = GREEN
    mossaic[1:h:2, 1:w:2] = img_array[1:h:2, 1:w:2, 2] # Bottom-Right = BLUE

    # Convert the numpy mosaic back to a PIL Image (Grayscale 'L')
    mossaic_img = Image.fromarray(mossaic, 'L')
    
    # Save as PNG
    mossaic_img.save(os.path.join(directory, name))
    print(f"Finished Bayer Pattern Image: {name}")

    return 0

def main():
    # --- Training Images ---
    print("----- Processing Training Images -----")
    for filename in os.listdir(DIV2K_TRAIN_HR):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        print(f"Started Processing: {filename}")
        try:
            # Gets Image using PIL, ensure it's RGB
            img_path = os.path.join(DIV2K_TRAIN_HR, filename)
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print(f"Error reading {filename}, skipping.")
            continue

        # Applies Bayer Filter (RGGB) to FULL-RES Image
        # and saves it directly to the INPUT dir
        bayer_img(img, filename, DIV2K_TRAIN_BAYER)
        print("----------")
        
    print("----- Finished Processing Training Images -----")
    print("")

    # --- Validation Images ---
    print("----- Processing Validation Images -----")
    for filename in os.listdir(DIV2K_VALID_HR):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        print(f"Started Processing: {filename}")
        try:
            # Gets Image using PIL, ensure it's RGB
            img_path = os.path.join(DIV2K_VALID_HR, filename)
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print(f"Error reading {filename}, skipping.")
            continue

        # Applies Bayer Filter (RGGB) to FULL-RES Image
        # and saves it directly to the INPUT dir
        bayer_img(img, filename, DIV2K_VALID_BAYER)
        print("----------")
        
    print("----- Finished Processing Validation Images -----")
    print("")

    return 0

# --- Run the script ---
if __name__ == "__main__":
    main()