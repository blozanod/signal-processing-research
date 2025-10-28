import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# Builds up on cnn.py by changing network architecture to ResNet
# Skips connections so that the gradient isn't just the output of convolutions:
# Output = Layers(input) + input, not just Layers(input)
# Allows for significantly deeper neural networks VGGs have limitation
# that deep neural networks (30+ layers) are worse due to jumbling data
# For experimentation purposes, initial accuracy at 10 epochs: 83.3%
# Improved accuracy to 87.3%, then 90.1%

# --------------------------------------------------------------------------------
# -- Data Loading --
# --------------------------------------------------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Add augmentations to the training transform
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 50% chance to flip the image horizontally
    transforms.RandomRotation(10),       # Rotate the image by up to 10 degrees
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

training_data = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform_train)

test_data = torchvision.datasets.CIFAR10(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform)

batch_size = 64

train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# --------------------------------------------------------------------------------
# -- Model Definition --
# --------------------------------------------------------------------------------

# Use CUDA
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path (no skips)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, we use a 1x1 conv to match them for the addition
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Adds input back into the output
        out = self.relu(out) # Now activates ReLU
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution layer (the "stem")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Stacked Residual Blocks (total of 6 convolution layers)
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2) # stride=2 downsamples the image
        self.layer3 = ResidualBlock(128, 256, stride=2) # stride=2 downsamples again
        self.layer3_block2 = ResidualBlock(256, 256, stride=1)
        self.layer4 = ResidualBlock(256, 512, stride=2) # stride=2 downsamples again
        self.layer4_block2 = ResidualBlock(512, 512, stride=1)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Averages each feature map to 1x1
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer3_block2(out)
        out = self.layer4(out)
        out = self.layer4_block2(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


model = ResNet().to(device)
print(model)

# --------------------------------------------------------------------------------
# -- Optimization --
# --------------------------------------------------------------------------------
epochs = 200

loss_fn = nn.CrossEntropyLoss()
# Momentum 'keeps direction' of gradient going to learn quicker
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning Rate Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# --------------------------------------------------------------------------------
# -- Checkpointing --
# --------------------------------------------------------------------------------
start_epoch = 0
CHECKPOINT_PATH = "model_checkpoint.pth"

try: # Try to load the model in, don't fail if the model doesn't exist yet
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch + 1}")
except FileNotFoundError:
    print("No checkpoint found. Starting training from scratch.")
except Exception as e:
    print(f"Error loading checkpoint: {e}. Starting from scratch.")

# --------------------------------------------------------------------------------
# -- Training & Testing Functions --
# --------------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # sets model to training mode
    for batch, (X, y) in enumerate(dataloader): # grabs a batch of images (size 4 in this case) and uses them
        X, y = X.to(device), y.to(device) # moves data to gpu for optimization

        # Compute prediction error (forward pass)
        pred = model(X) # gets the model's prediction
        loss = loss_fn(pred, y) # runs it through loss function to see how far off it is

        # Backpropagation
        loss.backward() # calculates the gradient of the loss, which then tells the optimizer where steepest
        # increase in loss is, allowing optimizer to adjust accordingly
        optimizer.step() # takes gradient and updates parameters in opposite direction (to minimize loss)
        optimizer.zero_grad() # resets gradients for next batch (garbage collection basically)

        # Prints the loss for the batch the model used
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # puts model on evaluation, not training mode
    test_loss, correct = 0, 0
    with torch.no_grad(): # removes gradients to optimize performance, as not needed for evaluation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Calculates batch loss and adds to running total
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Counts how many correct

    # Displays accuracy
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Runs model through epochs (generations)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    scheduler.step()

    # Checkpoints after each epoch
    # This saves your progress after every epoch.
    checkpoint = {
        'epoch': t + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved checkpoint for epoch {t + 1}")
print("Training Done!")

# This saves only the learned weights, which is all you need for making predictions.
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")