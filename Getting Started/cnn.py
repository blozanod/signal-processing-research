import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Experiments:
# Changed batch_size from 4 to 64, changed epochs from 2 to 5 (52.1% to 50.7%)
# Changed optimizer from SDG to the GOAT Adam (50.7% to 61.5%)
# Increased learning rate from 1e-3 to 5e-3 (61.5% to 61.6%)
# Added learning rate scheduler (61.6% to 61.4%)
# Increased epochs from 5 to 10 for lrs (61.4% to 61.4%)
# Redesigned NN to a VGG style approach (4 conv in 2 blocks, then linear):
# Dropped epochs back to 5, dropout to prevent overfit, much larger model

# --------------------------------------------------------------------------------
# -- Data Loading --
# --------------------------------------------------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = torchvision.datasets.CIFAR10(root='./data',
                                           train=True,
                                           download=True,
                                           transform=transform)

test_data = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------------------------------------------------------------------------------
# -- Model Definition --
# --------------------------------------------------------------------------------

# Use CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG-Style Network
        # Block 1: 32x32 img -> 16x16
        self.features1 = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Second convolution
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # turns 32x32 -> 16x16
        )

        # Block 2: 16x16 img -> 8x8
        self.features2 = nn.Sequential(
            # Third convolution
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Fourth convolution
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # turns 16x16 -> 8x8
        )

        # Flatten image and classify
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), # 128 from last conv2d, 8x8 because of img size
            nn.ReLU(),
            nn.Dropout(0.5), # to avoid overfitting
            nn.Linear(512, 10) # Leaves with 10 categories
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.classifier(x)
        return x

model = NeuralNetwork().to(device)
print(model)

# --------------------------------------------------------------------------------
# -- Optimization --
# --------------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# --------------------------------------------------------------------------------
# -- Checkpointing --
# --------------------------------------------------------------------------------
start_epoch = 0
CHECKPOINT_PATH = "cifar_model_checkpoint.pth"

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
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # moves data to gpu for optimization

        # Compute prediction error (forward pass)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Prints the loss for the batch the model used
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # puts model on evaluation, not training mode
    test_loss, correct = 0, 0
    with torch.no_grad(): # removes gradients to optimize performance
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Calculates batch loss and adds to running total
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Displays accuracy
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Runs model through epochs (generations)
epochs = 5
for t in range(start_epoch, epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    scheduler.step()

    # Checkpoints after each epoch
    checkpoint = {
        'epoch': t + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Saved checkpoint for epoch {t + 1}")
print("Training Done!")

# This saves only the learned weights, which is all you need for making predictions.
torch.save(model.state_dict(), "cifar_model.pth")
print("Saved PyTorch Model State to cifar_model.pth")