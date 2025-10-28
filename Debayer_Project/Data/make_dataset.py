import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# --------------------------------------------------------------------------------
# -- Image Dataset Class Definition --
# --------------------------------------------------------------------------------

class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform

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

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image