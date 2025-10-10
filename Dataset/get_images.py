import os
import cv2
import numpy as np
import requests

# Config
ACCESS_KEY = "bUCcR0m93j11PADyH0fhzQMjx9IjzY_VtrszyKNVJpc"
SECRET_KEY = "PNtoVgnj34hh196wjj0eMsOFXde95Y4Wkv3ClGxp7U0"
REQUEST_LIMIT = 50 # requests/hour, can be 5000 with production status
DOWNLOAD_PATH = 'Unsplash_Images'

IMAGE_COUNT = 200 # download image goal

API_URL = f"https://api.unsplash.com/photos/random"

# is_grayscale helper function
def is_grayscale(image_data_bytes):
    """
    Checks if an image is grayscale by analyzing its color saturation.
    This is more robust against JPEG compression artifacts than a direct channel comparison.

    Args:
        image_data_bytes (bytes): The raw byte content of the image.

    Returns:
        bool: True if the image is grayscale, False otherwise.
    """
    # You can adjust this threshold. Lower is stricter (closer to pure gray).
    # A good starting point is between 5 and 15.
    SATURATION_THRESHOLD = 10

    # Decode the image data from memory
    nparr = np.frombuffer(image_data_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return False # Cannot decode, assume it's not a valid image to check

    # Convert the image from BGR to HSV color space
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Calculate the average saturation value
    # The Saturation channel is the 2nd channel in HSV (index 1)
    avg_saturation = np.mean(img_hsv[:, :, 1])

    # If the average saturation is below the threshold, it's grayscale
    if avg_saturation < SATURATION_THRESHOLD:
        print(f"Detected grayscale image (avg saturation: {avg_saturation:.2f})")
        return True
    
    return False

# Request Count
NUM_REQUESTS = int(IMAGE_COUNT / REQUEST_LIMIT) + 1
download_count = 0

if (IMAGE_COUNT < REQUEST_LIMIT):
    REQUEST_LIMIT = IMAGE_COUNT

# Make download path
if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)

# Get Images Loop
for i in range(NUM_REQUESTS):
    if download_count > IMAGE_COUNT:
        break

    params = {
        'count': REQUEST_LIMIT,
        'client_id': ACCESS_KEY
    }

    # Make API Request
    try:
        response = requests.get(API_URL, params=params, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        photos = response.json()

        for photo in photos:
            # Ensure download count < goal
            if download_count >= IMAGE_COUNT:
                    break   
            
            # Create photo
            photo_id = photo['id'].encode('utf-8').hex()
            # Use the 'raw' URL
            photo_url = photo['urls']['raw']
            # Make photo file path
            file_path = os.path.join(DOWNLOAD_PATH, f"{photo_id}.jpg")
            
            if not os.path.exists(file_path):
                # Download the image content
                image_response = requests.get(photo_url, stream=True, timeout=15)
                image_response.raise_for_status()

                # Get raw image data
                image_data = image_response.content

                if is_grayscale(image_data):
                    print(f"Image {photo_id} is grayscale, skipping...")
                    continue

                # Save the image to a file
                with open(file_path, 'wb') as f:
                    for chunk in image_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                download_count += 1
                print("Finished downloading image number " + str(download_count))
            else:
                print("Image already in dataset, skipping...")
    
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")


print(f"\nDownload complete. {download_count} images saved in '{DOWNLOAD_PATH}'.")