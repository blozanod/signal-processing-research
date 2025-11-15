import torch
import torch.nn as nn
import torch.nn.functional as F

# Chose Gradient and not Perception (VGG) due to time to implement
class CharbonnierGradientLoss(nn.Module):
    # Charbonnier Loss: (Mix of L1 and L2 (MSE) Loss)
    # L(x,y) = sqrt((x-y)^2 + eps^2)
    #        = sqrt(diff^2 + eps^2)

    # Gradient Loss (Using Sobel filters):
    # Grad = grad_weight * abs(grad(x) - grad(y))

    # Combined:
    # L(x,y) = sqrt((x-y)^2 + eps^2) + [grad_weight * mag(grad(x) - grad(y))]

    def __init__(self, eps=1e-3, grad_weight=0.05):
        super().__init__()
        self.eps = eps
        self.grad_weight = grad_weight
        
        # Sobel filters (fixed, no grading)
        # detect edges and textures within the image
        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]], dtype=torch.float32)

        # Registers as buffers so DDP syncs them, otherwise error
        self.register_buffer("sobel_x", sobel_x.unsqueeze(1))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(1))

    # More robust than L1/L2, (L2 for small errors, L1 for large outliers)
    def charbonnier(self, diff):
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

    def forward(self, pred, target):
        # Base charbonnier (combination of L1 + L2)
        base = self.charbonnier(pred - target)
        
        num_channels = pred.shape[1]
        
        # Get current device (to move sobel to device)
        device = pred.device

        # Generates sobel filters (x -> horizontal edges, y -> vertical edges)
        sobel_x_filter = self.sobel_x.repeat(num_channels, 1, 1, 1).to(device)
        sobel_y_filter = self.sobel_y.repeat(num_channels, 1, 1, 1).to(device)

        # Applies sobel filters to prediction and target image via each channel
        # ie. applies filter to red channel, then green, then blue
        # Gradient loss looks at image sharpness + texture
        grad_pred_x = F.conv2d(pred, sobel_x_filter, padding=1, groups=num_channels)
        grad_pred_y = F.conv2d(pred, sobel_y_filter, padding=1, groups=num_channels)
        grad_tgt_x  = F.conv2d(target, sobel_x_filter, padding=1, groups=num_channels)
        grad_tgt_y  = F.conv2d(target, sobel_y_filter, padding=1, groups=num_channels)

        # Compute gradient charbonnier for both x,y dimensions on both target and prediction
        # this way grad_loss doesn't have the weakness of either L1 or L2
        grad_loss = (
            self.charbonnier(grad_pred_x - grad_tgt_x) +
            self.charbonnier(grad_pred_y - grad_tgt_y)
        ) * 0.5  # average x and y

        # Final loss is sum of weighted gradient and charbonnier loss
        return base + self.grad_weight * grad_loss