import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torchvision.transforms.functional import crop
import os
import random
import numpy as np

# Amount dataset is enhanced by:
enhanced = 100

# Random log uniform sampling function for color shifting
def rand_log_uniform(low, high):
    return float(np.exp(np.random.uniform(np.log(low), np.log(high))))

class ImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform, crop_size_raw, is_train=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.crop_size_raw = crop_size_raw
        self.is_train = is_train

        # Ensure inputs & outputs match
        self.input_filenames = sorted(os.listdir(input_dir))
        self.output_filenames = sorted(os.listdir(output_dir))

        if is_train:
            self.input_filenames = self.input_filenames * enhanced
            self.output_filenames = self.output_filenames * enhanced

        # Ensure same number of images
        assert len(self.input_filenames) == len(self.output_filenames), "Input and Output directories have different length"

        self.to_tensor = transforms.ToTensor()

        # Cache all images to RAM for efficiency
        self.cached_inputs = []
        self.cached_outputs = []

        unique_length = len(self.input_filenames) // enhanced if is_train else len(self.input_filenames)
        
        for i in range(unique_length):
            input_path = os.path.join(self.input_dir, self.input_filenames[i])
            output_path = os.path.join(self.output_dir, self.output_filenames[i])

            # Read images
            input_image = torchvision.io.read_image(input_path)
            output_image = torchvision.io.read_image(output_path)

            # Add to cache
            self.cached_inputs.append(input_image)
            self.cached_outputs.append(output_image)

    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, index):
        # Calculate adjusted index to retrieve image
        base_len = len(self.cached_inputs)
        adj_index = index % base_len

        # Retrieve cached images
        input_image = self.cached_inputs[adj_index]
        output_image = self.cached_outputs[adj_index]

        # input_image is [1,H,W] if grayscale file, otherwise take one channel
        if input_image.shape[0] > 1:
            input_image = input_image[0:1]

        # Random Crop
        crop = self.crop_size_raw
        H, W = input_image.shape[1], input_image.shape[2]

        if self.is_train:
            # Random top-left
            i = random.randint(0, H - crop)
            j = random.randint(0, W - crop)
        else:
            # Consistent center crop
            i = (H - crop) // 2
            j = (W - crop) // 2
        # Ensure crop has even starts for packing
        i = (i // 2) * 2
        j = (j // 2) * 2

        # Crops full size images
        inp = input_image[:, i:i+crop, j:j+crop]  # [1,c,c]
        out = output_image[:, i:i+crop, j:j+crop]  # [3,c,c]

        # Pack RGGB bayer pattern
        b = inp[0]
        R  = b[0::2, 0::2]
        G1 = b[0::2, 1::2]
        G2 = b[1::2, 0::2]
        B  = b[1::2, 1::2]
        packed = torch.stack([R, G1, G2, B], dim=0) # [4, c/2, c/2], uint8

        # Convert to float in [0,1]
        input_tensor = packed.float().div_(255.0)
        output_tensor = out.float().div_(255.0)

        # Random flip
        if self.is_train:
            if random.random() > 0.5:
                input_tensor = input_tensor.flip(-1)
                output_tensor = output_tensor.flip(-1)
            if random.random() > 0.5:
                input_tensor = input_tensor.flip(-2)
                output_tensor = output_tensor.flip(-2)

        # Random color shift
        """
        if self.is_train and (random.random() < 0.9):
            gR = rand_log_uniform(0.9, 1.1)
            gG = rand_log_uniform(0.9, 1.1)
            gB = rand_log_uniform(0.9, 1.1)
            exp_gain = rand_log_uniform(0.9, 1.1)

            input_tensor[0].mul_(gR * exp_gain)
            input_tensor[1].mul_(gG * exp_gain)
            input_tensor[2].mul_(gG * exp_gain)
            input_tensor[3].mul_(gB * exp_gain)

            output_tensor[0].mul_(gR * exp_gain)
            output_tensor[1].mul_(gG * exp_gain)
            output_tensor[2].mul_(gB * exp_gain)

            input_tensor.clamp_(0.0, 1.0)
            output_tensor.clamp_(0.0, 1.0)
        """

        return input_tensor, output_tensor