import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Distributed Data Parallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Dataset and UNet Scripts
from Data.make_dataset import ImageDataset
from Model.unet import UNet

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 
DATA_DIR = os.path.join(project_root,"Data","Final_Dataset_Images")
INPUT_DIR = os.path.join(DATA_DIR,"Train","Input")
OUTPUT_DIR = os.path.join(DATA_DIR,"Train","Target")
VALID_INPUT_DIR = os.path.join(DATA_DIR,"Validate","Input")
VALID_OUTPUT_DIR = os.path.join(DATA_DIR,"Validate","Target")

# --------------------------------------------------------------------------------
# -- Training & Testing Functions --
# --------------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, device):
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

def validate(dataloader, model, loss_fn, device):
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

# --------------------------------------------------------------------------------
# -- DDP Functions --
# --------------------------------------------------------------------------------
def setup():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup():
    dist.destroy_process_group()

def main():
    # --------------------------------------------------------------------------------
    # -- Data Loading --
    # --------------------------------------------------------------------------------
    BATCH_SIZE = 8

    # Transforms
    TRAIN_IMG_SIZE = (256, 256)

    transform = transforms.Compose([
    transforms.Resize(TRAIN_IMG_SIZE),
    transforms.ToTensor(),
    ])

    # Create train dataset and dataloader
    train_dataset = ImageDataset(INPUT_DIR, OUTPUT_DIR, transform)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=train_sampler)
    print(f"Training Data: Found {len(train_dataset)} image pairs.")

    # Create validation dataset and dataloader
    valid_dataset = ImageDataset(VALID_INPUT_DIR, VALID_OUTPUT_DIR, transform)
    valid_sampler = DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=valid_sampler)
    print(f"Validation Data: Found {len(valid_dataset)} image pairs.")

    # --------------------------------------------------------------------------------
    # -- Model --
    # --------------------------------------------------------------------------------
    # Use CUDA
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = UNet().to(device)
    model = DDP(model, device_ids=[device])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
        train_sampler.set_epoch(t)
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        validate(valid_dataloader, model, loss_fn, device)
        scheduler.step()

        # Checkpoints after each epoch
        # This saves your progress after every epoch.
        checkpoint = {
            'epoch': t + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if dist.get_rank() == 0:
            torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Saved checkpoint for epoch {t + 1}")
    print("Training Done!")

    # This saves only the learned weights, which is all you need for making predictions.
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


if __name__ == "__main__":
    setup()
    main()
    cleanup()