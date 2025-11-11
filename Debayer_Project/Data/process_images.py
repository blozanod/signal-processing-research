# OK to leave like this since only going to be running once
# Ideally, implement parallel processing and optimizing memory loading and usage
# will significantly reduce runtime (10-15 mins to 1 or 2)

import cv2
import numpy as np
import os

M = 10 # Vertical Cells
N = 10 # Horizontal Cells

# Current Filepath
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# DIV 2K
DIV2K_DIR = os.path.join(CURRENT_PATH, "DIV2K_Images")
DIV2K_TRAIN_DIR = os.path.join(DIV2K_DIR, "DIV2K_Train_HR")
DIV2K_VALID_DIR = os.path.join(DIV2K_DIR, "DIV2K_Valid_HR")
DIV2K_TRAIN_BAYER_DIR = os.path.join(DIV2K_DIR, "DIV2K_Train_Bayer")
DIV2K_VALID_BAYER_DIR = os.path.join(DIV2K_DIR, "DIV2K_Valid_Bayer")

# Final Dataset
FINAL_DATASET_DIR = os.path.join(CURRENT_PATH, "Final_Dataset_Images")
FINAL_TRAIN_DIR = os.path.join(FINAL_DATASET_DIR, "Train")
TRAIN_TARGET_DIR = os.path.join(FINAL_TRAIN_DIR, "Target")
TRAIN_INPUT_DIR = os.path.join(FINAL_TRAIN_DIR, "Input")
FINAL_VALIDATE_DIR = os.path.join(FINAL_DATASET_DIR, "Validate")
VALIDATE_TARGET_DIR = os.path.join(FINAL_VALIDATE_DIR, "Target")
VALIDATE_INPUT_DIR = os.path.join(FINAL_VALIDATE_DIR, "Input")


# Bayer pattern algorithm
def bayer_img(img, name, directory):
    # print("Creating Bayer Pattern Image: " + str(name))
    if os.path.exists(os.path.join(directory,name)):
        print("Image previously processed, skipping...")
        return 0
    
    h,w, channels = img.shape

    mossaic = np.zeros((h, w), dtype=img.dtype)

    mossaic[0:h:2, 0:w:2] = img[0:h:2, 0:w:2, 2] # red
    mossaic[1:h:2, 0:w:2] = img[1:h:2, 0:w:2, 1] # green
    mossaic[0:h:2, 1:w:2] = img[0:h:2, 1:w:2, 1] # green
    mossaic[1:h:2, 1:w:2] = img[1:h:2, 1:w:2, 0] # blue

    # Save as PNG
    cv2.imwrite(os.path.join(directory, name), mossaic)
    print("Finished Bayer Pattern Image: " + name)

    return 0

# Split image into chunks
def split_img(img, name, directory):
    h,w,channels = img.shape
    #print("Splitting Image: " + str(name))
    #print("Height: " + str(h) + ", Width: " + str(w) + ", Channels: ", str(channels))

    # Divide image into a MxN grid
    h_step = h//M
    w_step = w//N

    for i in range(M):
        for j in range(N):
            # New filename
            filename = str(i) + str(j) + "_" + name
            if os.path.exists(os.path.join(directory,filename)):
                continue

            # Select image chunk
            top = i * h_step
            bottom = (i+1) * h_step
            left = j * w_step
            right = (j+1) * w_step

            chunk = img[top:bottom, left:right]
            
            # Write image to directory
            cv2.imwrite(os.path.join(directory,filename),chunk)    
    
    print("Finished Splitting Image" + str(name))
    return 0

def main():
    # Training Images
    print("----- Processing Training Images -----")
    for filename in os.listdir(DIV2K_TRAIN_DIR):
        print("Started Processing: " + str(filename))
        # Gets Image
        img = cv2.imread(os.path.join(DIV2K_TRAIN_DIR, filename))
        if img is None:
            raise ValueError("Image not found or could not be read.")
        
        # Splits Unmodified Image (Target)
        split_img(img, filename, TRAIN_TARGET_DIR)
        print("----------")

        # Applies Bayer Filter to Whole Image
        bayer_img(img, filename, DIV2K_TRAIN_BAYER_DIR)
        print("----------")

        # Gets Bayer Image
        img = cv2.imread(os.path.join(DIV2K_TRAIN_BAYER_DIR, filename))
        if img is None:
            raise ValueError("Image not found or could not be read.")
        split_img(img, filename, TRAIN_INPUT_DIR)
        print("----------")
        
    print("----- Finished Processing Training Images -----")
    print("")

    # Validation Images
    print("----- Processing Validation Images -----")
    for filename in os.listdir(DIV2K_VALID_DIR):
        print("Started Processing: " + str(filename))
        # Gets Image
        img = cv2.imread(os.path.join(DIV2K_VALID_DIR, filename))
        if img is None:
            raise ValueError("Image not found or could not be read.")
        
        # Splits Unmodified Image (Target)
        split_img(img, filename, VALIDATE_TARGET_DIR)
        print("----------")

        # Applies Bayer Filter to Whole Image
        bayer_img(img, filename, DIV2K_VALID_BAYER_DIR)
        print("----------")

        # Gets Bayer Image
        img = cv2.imread(os.path.join(DIV2K_VALID_BAYER_DIR, filename))
        if img is None:
            raise ValueError("Image not found or could not be read.")
        split_img(img, filename, VALIDATE_INPUT_DIR)
        print("----------")
        
    print("----- Finished Processing Validation Images -----")
    print("")

    return 0

main()