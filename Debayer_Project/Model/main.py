import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


from Data import make_dataset as md
from Model import unet as un

# --------------------------------------------------------------------------------
# -- First Testing Results --
# --------------------------------------------------------------------------------
# Epoch 1: Validation Avg Loss: 0.007502
# Epoch 2: Validation Avg Loss: 0.006270
# Epoch 3: Validation Avg Loss: 0.002309
# Epoch 4: Validation Avg Loss: 0.001356
# Therefore, it is learning, and becoming more accurate, will validate with visual evidence
# See visual.py for visual evidence

# --------------------------------------------------------------------------------
# -- Training & Testing Functions --
# --------------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    
    # Wrap the dataloader with tqdm
    # leave=False means the bar disappears after the loop is done
    loop = tqdm(dataloader, desc="  Training", leave=False)
    
    for batch, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device) 

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update the progress bar's description with the current loss
        loop.set_postfix(loss=loss.item())

def validate(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval() 
    test_loss = 0
    
    # Wrap the dataloader with tqdm
    loop = tqdm(dataloader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for X, y in loop: # 2. Iterate over the tqdm loop
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            batch_loss = loss_fn(pred, y).item()
            test_loss += batch_loss

            # 3. Update the progress bar with the running average loss
            # loop.n is the current batch number (starting from 0)
            running_avg_loss = test_loss / (loop.n + 1)
            loop.set_postfix(avg_loss=f"{running_avg_loss:.4f}")

    test_loss /= num_batches
    print(f"Validation Error: \n Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # --------------------------------------------------------------------------------
    # -- Data Loading --
    # --------------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 


    DATA_DIR = os.path.join(project_root, "Data", "Final_Dataset_Images")
    INPUT_DIR = os.path.join(DATA_DIR,"Train\\Input")
    OUTPUT_DIR = os.path.join(DATA_DIR,"Train\\Target")
    VALID_INPUT_DIR = os.path.join(DATA_DIR,"Validate\\Input")
    VALID_OUTPUT_DIR = os.path.join(DATA_DIR,"Validate\\Target")

    BATCH_SIZE = 8

    # Transforms
    TRAIN_IMG_SIZE = (256, 256)

    transform = transforms.Compose([
    transforms.Resize(TRAIN_IMG_SIZE),
    transforms.ToTensor(),
    ])

    # Create train dataset and dataloader
    train_dataset = md.ImageDataset(INPUT_DIR, OUTPUT_DIR, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print(f"Training Data: Found {len(train_dataset)} image pairs.")

    # Create validation dataset and dataloader
    valid_dataset = md.ImageDataset(VALID_INPUT_DIR, VALID_OUTPUT_DIR, transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Validation Data: Found {len(valid_dataset)} image pairs.")

    # --------------------------------------------------------------------------------
    # -- Model --
    # --------------------------------------------------------------------------------
    # Use CUDA
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = un.UNet().to(device)
    print(model)

    # --------------------------------------------------------------------------------
    # -- Optimization --
    # --------------------------------------------------------------------------------
    epochs = 1

    loss_fn = nn.MSELoss()
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

    # Runs model through epochs (generations)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        validate(valid_dataloader, model, loss_fn)
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