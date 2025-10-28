import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import StepLR

# Experiments:
# Ran as pytorch tutorial instructed (64.7% accuracy)
# Increased epoch count from 5 to 10 (increased accuracy from 64.7% to 70.7%)
# Changed learning rate from 1e-3 to 5e-3 (increased accuracy from 70.7% to 82.3%)
# Changed optimizer from SGD to Adam (reset lr to 1e-3) (increased accuracy from 82.3% to 87.3%)
# Added learning rate scheduler (increqased accuracy from 87.3% to 89.1%)
# Increased capacity to 1024, added dropout, added bottleneck (decreased accuracy to 87.7% but good learning)

# Download training data from open datasets.
# FashionMINST is training data set for AI to sort clothing based on types of clothing
# eg. short vs long sleeve shirts, dresses, heels, shoes... has 10 categories
# Training data is what is used to train the model
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
# Test data is data the model has never seen that it will be tested on at the end
# it tests to see if model can generalize
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
# Data loaders wrap around the dataset and transform it into an object that can be iterated through
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Model Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    # Defines what the layers are
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # Flattens each image from 2D array into 3D array
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024), # First "hidden layer" with 28*28 = 784 inputs and 512 outputs
            nn.ReLU(), # Activation function that introduces non linearity so network can learn complex relations
            # Regularization
            nn.Dropout(0.5),
            nn.Linear(1024, 1024), # Second hidden layer that takes in previous output as input, gives 512 more
            nn.ReLU(),
            # Regularization
            nn.Dropout(0.5),
            nn.Linear(1024, 10) # Output layer, gives 10 outputs because FashionMINST has 10 classes
        )

    # Defines how data flows through layers
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) # Has data pass through layers to make a tensor of 10 logits
        # Logits:
        # The "raw scores" for how confident the model is in each category
        # represents how confident the model is that the image fits into each category,
        # high score = high confidence. These are the outputs from the output layer
        return logits

model = NeuralNetwork().to(device)
print(model)

# Model Optimization
loss_fn = nn.CrossEntropyLoss() # Measures model error
# the nn function measures how far off the model is from the right answer, high number = far off
# CrossEntropyLoss is standard for multiple category datasets
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adjusts model weights and balanced to minimize loss
# it is like the tutor while the loss_fn is the test score

# Learning rate scheduler:
# This will reduce the LR by a factor of 0.1 every 3 epochs
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Loading the model in
# This part will load the last saved state, so you can resume training.
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

# Model training: One full loop of training (or generation)
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # sets model to training mode
    for batch, (X, y) in enumerate(dataloader): # grabs a batch of images (size 64 in this case) and uses them
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

# Tests model on data it has never seen before
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

# Runs model through 10 epochs (generations)
epochs = 10
for t in range(start_epoch, epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

    scheduler.step() # Steps learning rate scheduler

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