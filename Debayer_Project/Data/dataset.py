import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform, crop_size, is_train=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.crop_size = crop_size
        self.is_train = is_train

        # Ensure inputs & outputs match
        self.input_filenames = sorted(os.listdir(input_dir))
        self.output_filenames = sorted(os.listdir(output_dir))

        if is_train:
            self.input_filenames = self.input_filenames * 10
            self.output_filenames = self.output_filenames * 10

        # Ensure same number of images
        assert len(self.input_filenames) == len(self.output_filenames), "Input and Output directories have different length"

    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, index):
        adj_index = index % (len(self.input_filenames) // 10) # adjusted index for increasing image dataset size

        input_path = os.path.join(self.input_dir, self.input_filenames[adj_index])
        output_path = os.path.join(self.output_dir, self.output_filenames[adj_index])

        input_image = Image.open(input_path).convert("L")
        output_image = Image.open(output_path).convert("RGB")

        # Random Crop
        if self.is_train:
            # Get parameters for a RANDOM crop
            i, j, h, w = transforms.RandomCrop.get_params(
                input_image, output_size=(self.crop_size, self.crop_size))
        else:
            # Get parameters for a CENTER crop
            w_img, h_img = input_image.size 
            h_crop = self.crop_size
            w_crop = self.crop_size
            
            i = (h_img - h_crop) // 2 # 'i' is the top offset
            j = (w_img - w_crop) // 2 # 'j' is the left offset
            h = h_crop
            w = w_crop

        # Random flip
        if self.is_train:
            # Horizontal
            if random.random() > 0.5:
                input_image = F.hflip(input_image)
                output_image = F.hflip(output_image)
            # Vertical
            if random.random() > 0.5:
                input_image = F.vflip(input_image)
                output_image = F.vflip(output_image)

        # Unpack input image to 4 channels (R, G1, G2, B)
        # lets model know what pixels are red, green, and blue
        bayer_array = np.array(input_image) 
        h_full, w_full = bayer_array.shape

        channel_r = np.zeros_like(bayer_array)
        channel_g1 = np.zeros_like(bayer_array)
        channel_g2 = np.zeros_like(bayer_array)
        channel_b = np.zeros_like(bayer_array)

        channel_r[0::2, 0::2] = bayer_array[0::2, 0::2]  # Red
        channel_g1[0::2, 1::2] = bayer_array[0::2, 1::2] # Green 1
        channel_g2[1::2, 0::2] = bayer_array[1::2, 0::2] # Green 2
        channel_b[1::2, 1::2] = bayer_array[1::2, 1::2]  # Blue

        # Stack Image
        input_stacked = np.stack([channel_r, channel_g1, channel_g2, channel_b], axis=-1)

        # Applies crop to image
        input_cropped = input_stacked[i:i+h, j:j+w, :] # Numpy Crop
        output_image = F.crop(output_image, i, j, h, w)

        input_tensor = self.transform(input_cropped)
        output_tensor = self.transform(output_image)

        return input_tensor, output_tensor