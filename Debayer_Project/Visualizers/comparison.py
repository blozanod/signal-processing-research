import os
from PIL import Image

def create_comparison(id_base, original_path, version_path, output_path, crop_size=64, scale=10):
    """
    Crops two images, scales them up (pixelated), and stitches them side-by-side.
    """
    try:
        # Load images
        img_orig = Image.open(original_path)
        img_ver = Image.open(version_path)
        
        # Helper function to crop center
        def get_center_crop(img, w, h):
            iw, ih = img.size
            left = (iw - w) / 2
            top = (ih - h) / 2
            right = (iw + w) / 2
            bottom = (ih + h) / 2
            return img.crop((left, top, right, bottom))

        # 1. Crop Center 64x64
        crop_orig = get_center_crop(img_orig, crop_size, crop_size)
        crop_ver = get_center_crop(img_ver, crop_size, crop_size)

        # 2. Resize with Nearest Neighbor (Hard Pixels)
        # This locks in the "pixelated" look so PowerPoint can't blur it.
        new_size = (crop_size * scale, crop_size * scale)
        resize_orig = crop_orig.resize(new_size, resample=Image.NEAREST) 
        resize_ver = crop_ver.resize(new_size, resample=Image.NEAREST)

        # 3. Create a canvas for side-by-side
        # Width = 2 images + 20px padding
        padding = 20
        total_width = new_size[0] * 2 + padding
        total_height = new_size[1]
        
        # Create white background
        combined_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
        
        # Paste them in: Original on Left, Version on Right
        combined_img.paste(resize_orig, (0, 0))
        combined_img.paste(resize_ver, (new_size[0] + padding, 0))

        # 4. Save
        combined_img.save(output_path)
        print(f"Generated comparison: {output_path}")

    except FileNotFoundError as e:
        print(f"Skipping {id_base}: File not found ({e.filename})")
    except Exception as e:
        print(f"Error processing {id_base}: {e}")

def main():
    # --- CONFIGURATION ---
    target_ids = ["0801", "0802", "0808", "0844", "0852", "0873", "0898"]
    output_folder_name = "comparisons"
    crop_size = 64
    scale_factor = 10  # 64px * 10 = 640px final size per image block
    # ---------------------

    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, output_folder_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Creating Side-by-Side Comparisons ---")

    for file_id in target_ids:
        # Define input filenames
        orig_filename = f"{file_id}.png"
        ver_filename = f"{file_id}v11.png"
        
        # Define full paths
        path_orig = os.path.join("../images", orig_filename)
        path_ver = os.path.join("../images", ver_filename)
        
        # Output filename (e.g., "0801_compare.png")
        path_out = os.path.join(output_dir, f"{file_id}v11_compare.png")
        
        create_comparison(file_id, path_orig, path_ver, path_out, crop_size, scale_factor)

    print("--- Done ---")

if __name__ == "__main__":
    main()