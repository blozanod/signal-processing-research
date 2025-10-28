import cv2
import os
import numpy as np

def debayer_images_in_directory():
    """
    Finds all lossless images in the 'Bayer_Images' directory,
    applies a debayering algorithm, and saves the result as a JPEG
    in the same directory.
    """
    # --- 1. Define Paths ---
    # Get the absolute path of the directory where this script is located
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full path to the Bayer_Images directory
    bayer_dir = os.path.join(current_path, "Bayer_Images")

    # Check if the directory exists
    if not os.path.isdir(bayer_dir):
        print(f"Error: Directory not found at '{bayer_dir}'")
        print("Please make sure the 'Bayer_Images' directory is in the same folder as this script.")
        return

    print(f"Searching for images in: {bayer_dir}")

    # --- 2. Loop Through Files and Process ---
    for filename in os.listdir(bayer_dir):
        # Process only lossless image formats like PNG or TIFF for accurate results
        if filename.lower().endswith(('.png', '.tiff', '.bmp')):
            
            input_path = os.path.join(bayer_dir, filename)
            print(f"\nProcessing: {filename}")

            # --- 3. Load the Bayer Image ---
            # It is CRITICAL to load it as a single-channel grayscale image
            bayer_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if bayer_image is None:
                print(f"  - Could not read image. Skipping.")
                continue

            # --- 4. Debayer the Image ---
            # We assume an RGGB Bayer pattern, where the top-left pixel is Red.
            # cv2.COLOR_BAYER_RG2BGR is the corresponding conversion code.
            # 
            try:
                debayered_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_RG2BGR)
                print(f"  - Successfully debayered image.")
            except cv2.error as e:
                print(f"  - OpenCV error during debayering: {e}. Skipping.")
                continue

            # --- 5. Export as JPEG ---
            # Define the output filename by changing the extension to .jpg
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}_debayered.jpg"
            output_path = os.path.join(bayer_dir, output_filename)

            # Save the resulting 3-channel color image
            cv2.imwrite(output_path, debayered_image)
            print(f"  - Saved result as: {output_filename}")

    print("\nProcessing complete.")

# This block ensures the script runs when executed directly
if __name__ == "__main__":
    debayer_images_in_directory()