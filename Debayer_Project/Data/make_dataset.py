import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# --------------------------------------------------------------------------------
# -- Image Dataset Class Definition --
# --------------------------------------------------------------------------------

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

        # Ensure same number of images
        assert len(self.input_filenames) == len(self.output_filenames), "Input and Output directories have different length"

    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.input_filenames[index])
        output_path = os.path.join(self.output_dir, self.output_filenames[index])

        input_image = Image.open(input_path).convert("L")
        output_image = Image.open(output_path).convert("RGB")

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
        
        # Applies crop to image
        input_image = F.crop(input_image, i, j, h, w)
        output_image = F.crop(output_image, i, j, h, w)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image