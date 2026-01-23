import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CharbonnierLoss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, eps=1e-3, reduction="mean"):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, X, Y):
        diff = X - Y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
    
class EdgeLoss(nn.Module):
    """Edge loss using Sobel filters."""
    def __init__(self, reduction="mean"):
        super(EdgeLoss, self).__init__()
        self.reduction = reduction

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]])
        
        # Reshape filters for RGB images
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        # Compute gradients
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)

        # Compute L1 loss on gradients
        loss_x = F.l1_loss(grad_pred_x, grad_target_x, reduction=self.reduction)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y, reduction=self.reduction)

        return loss_x + loss_y

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 feature maps."""
    def __init__(self, layer_weights=None, reduction="mean"):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:23].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Buffer VGG Layers so they move to GPU
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Select layers for feature extraction: 
        if layer_weights is None:
            self.layer_weights = {
                '3': 1.0,   # relu1_2
                '8': 1.0,   # relu2_2
                '15': 1.0,  # relu3_3
                '22': 1.0   # relu4_3
            }
        else:
            self.layer_weights = layer_weights
        
        self.reduction = reduction

    def forward(self, pred, target):
        # Normalize inputs to match VGG training stats
        # VGG expects [0,1] images normalized by ImageNet mean/std
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        loss = 0.0
        x = pred_norm
        y = target_norm
        
        # Extract features layer by layer
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            y = layer(y)
            
            if name in self.layer_weights:
                # Calculate L1 loss for this feature scale
                layer_loss = F.l1_loss(x, y, reduction=self.reduction)
                loss += self.layer_weights[name] * layer_loss
        
        return loss
    
class CombinedLoss(nn.Module):
    """Combine edge, perceptual, and Charbonnier losses."""
    def __init__(self, edge_weight=0.1, perceptual_weight=0.05, charbonnier_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight
        self.charbonnier_weight = charbonnier_weight

        # Calculate each loss component
        self.edge_loss = EdgeLoss()
        self.perceptual_loss = PerceptualLoss()
        self.charbonnier_loss = CharbonnierLoss()

    def forward(self, pred, target):
        loss = (
            self.edge_weight * self.edge_loss(pred, target) +
            self.perceptual_weight * self.perceptual_loss(pred, target) +
            self.charbonnier_weight * self.charbonnier_loss(pred, target)
        )
        return loss