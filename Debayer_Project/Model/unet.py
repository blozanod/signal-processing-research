import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# -- Model Definition --
# --------------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path (no skips)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out) # Now activates ReLU
        return out

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Stacked DoubleConv Blocks (total of 8 convolution layers)
        # Down Layers
        self.down1 = DoubleConv(1, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)

        # Bottleneck Layer
        self.bottleneck = DoubleConv(512,512, stride=1)
        
        # Up Layers
        self.up_conv1 = nn.ConvTranspose2d(512, 512, stride=2, kernel_size=2)
        self.up1 = DoubleConv(1024,256)

        self.up_conv2 = nn.ConvTranspose2d(256, 256, stride=2, kernel_size=2)
        self.up2 = DoubleConv(512,128)

        self.up_conv3 = nn.ConvTranspose2d(128, 128, stride=2, kernel_size=2)
        self.up3 = DoubleConv(256,64)

        self.up_conv4 = nn.ConvTranspose2d(64, 64, stride=2, kernel_size=2)
        self.up4 = DoubleConv(128,32)

        self.up_conv5 = nn.ConvTranspose2d(32, 32, stride=2, kernel_size=2)
        self.up5 = DoubleConv(64,32)

        # Final Layer
        self.output = nn.Conv2d(32, 3, kernel_size=1, stride=1)

        # Pooling
        self.maxpool = nn.MaxPool2d((2, 2)) # Averages each feature map to 1x1

        self.sigmoid = nn.Sigmoid() # Squashes values from 0-1 to avoid oversaturation/blown out images

        # Dropout
        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x):
        # Padding so every image size works
        _, _, H, W = x.shape
        # Calculate the padding needed to make H and W multiples of 32
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        # Apply padding
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        # Down (Encoder)
        skip1 = self.down1(x_padded)
        pool1 = self.maxpool(skip1)

        skip2 = self.down2(pool1)
        pool2 = self.maxpool(skip2)

        skip3 = self.down3(pool2)
        pool3 = self.maxpool(skip3)

        skip4 = self.down4(pool3)
        pool4 = self.maxpool(skip4)

        skip5 = self.down5(pool4)
        pool5 = self.maxpool(skip5)

        # Bottleneck
        bottleneck = self.bottleneck(pool5)
        bottleneck = self.dropout(bottleneck)

        # Up (Decoder)
        up_conv1 = self.up_conv1(bottleneck)
        concat1 = torch.cat([up_conv1, skip5], dim=1)
        up1 = self.up1(concat1)

        up_conv2 = self.up_conv2(up1)
        concat2 = torch.cat([up_conv2, skip4], dim=1)
        up2 = self.up2(concat2)

        up_conv3 = self.up_conv3(up2)
        concat3 = torch.cat([up_conv3, skip3], dim=1)
        up3 = self.up3(concat3)

        up_conv4 = self.up_conv4(up3)
        concat4 = torch.cat([up_conv4, skip2], dim=1)
        up4 = self.up4(concat4)

        up_conv5 = self.up_conv5(up4)
        concat5 = torch.cat([up_conv5, skip1], dim=1)
        up5 = self.up5(concat5)

        # Final Layer
        out_padded = self.output(up5)

        # Crop out layer
        out = out_padded[:, :, :H, :W]

        # Squashed
        out = self.sigmoid(out)

        return out
    
# --- Test ---
# model = UNet()
# [Batch, Channels, Height, Width]
# dummy_input = torch.randn(4, 1, 204, 140) 
# output = model(dummy_input)

# print("Input shape:", dummy_input.shape)
# print("Output shape:", output.shape) # Should be same as in, but with 3 channels