"""
Contains functions for training and testing a class-conditioned flow matching model for MNIST digit generation.
"""

import argparse
import wandb
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from models.flow_matcing_models import *
from models.denoisers import *
from data_setup import *

def train_one_epoch(model: FlowMatchingClassCond,
                    trainloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    wandb_logging: bool = False):
    
    """Trains model for one epoch and returns average loss.
    
    For each batch of sample images, runs forward pass to compute losses,
    performs gradient computation, and takes gradient step. Prints average
    loss over every 100 mini-batches.

    Args:
        model: The class-conditioned flow matching model to be trained.
        trainloader: Dataloader instance containing training MNIST images.
        optimizer: PyTorch optimizer
        device: Device model sits on
        wandb_logging: bool, whether to log training steps in wandb, False by default
    
    Returns:
        None
    """

    running_loss = 0

    for i, (imgs, classes) in enumerate(trainloader):

        # Move images and classes to target device
        imgs = imgs.to(device)
        classes = classes.to(device)

        # Run forward method to get losses
        loss = model(imgs, classes)
        if wandb_logging:
            wandb.log({"loss": loss})

        # Zero gradients
        optimizer.zero_grad()

        # Backpropagate loss
        loss.backward()

        # Take gradient step
        optimizer.step()

        running_loss += loss

        if i and i % 100 == 0:
            print(f"Average Loss From Batch {i-100}-{i}: {running_loss / 100}")
            running_loss = 0


def train(model: FlowMatchingClassCond,
          trainloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          device: torch.device,
          num_epochs: int,
          target_dir: Path,
          wandb_logging: bool = False):
    
    """Runs training over num_epochs epochs.

    Invokes train_one_epoch num_epochs times, and concatenates list of losses
    per step. Returns list of losses.

    Args:
        model: PyTorch flow matching model.
        trainloader: Dataloader instance with sample images.
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        device: Device model sits on
        num_epochs: int, number of epochs to train for
        target_dir: Directory in which to save model checkpoints
        wandb_logging: bool, whether to log training steps in wandb, False by default
    
    Returns:
        None
    """

    # Move model to device
    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        print(f"STARTING EPOCH {epoch+1}")
        train_one_epoch(model,
                        trainloader,
                        optimizer,
                        device,
                        wandb_logging)
        scheduler.step()
        model_path = target_dir / f"epoch_{epoch+1}.pth"
        torch.save(obj=model.state_dict(),
                   f=model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    training_details = parser.add_argument_group("Hyperparameters detailing training procedure")
    training_details.add_argument(
        "--epochs",
        type=int,
        default=5
    )
    training_details.add_argument(
        "--batch-size",
        type=int,
        default=64
    )
    training_details.add_argument(
        "--lr",
        type=float,
        default=1e-2
    )
    training_details.add_argument(
        "--scheduler-multiplier",
        type=float,
        default=0.1
    )
    training_details.add_argument(
        "--p-uncond",
        type=float,
        default=0.2
    )

    unet_architecture = parser.add_argument_group("Hyperparameters for UNet Architecture")
    unet_architecture.add_argument(
        "--input-dim",
        type=int,
        default=1
    )
    unet_architecture.add_argument(
        "--hidden-dim",
        type=int,
        default=64
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default='',
        help='Indicate a WandB project to enable logging of loss during training steps.'
    )

    args = parser.parse_args()
    
    # Get dataset
    transform = transforms.Compose([ToTensor()])
    trainset = datasets.MNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transform)
    
    # Create Dataloader
    trainloader = DataLoader(trainset, batch_size=args.batch_size)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU is available.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Mac M4 GPU.")
    else:
        device = "cpu"
        print("No GPU available, using CPU.")

    # Instantiate model, optimizer, and scheduler
    class_cond_unet = ClassConditionalUNet(args.input_dim, len(trainset.classes), args.hidden_dim)
    class_cond_flow_matching = FlowMatchingClassCond(class_cond_unet,
                                                     device=device,
                                                     p_uncond=args.p_uncond,
                                                     ).to(device)
    optimizer = torch.optim.Adam(class_cond_flow_matching.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_multiplier**(1/args.epochs))

    # Instantiate model path
    target_dir = Path("./weights")
    target_dir = target_dir / "ClassConditionedFlowMatching" / (f"timestamp_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    target_dir.mkdir(parents=True,
                         exist_ok=True)

    
    # Initialize WandB logging
    if args.wandb_project:
        wandb.login()
        run = wandb.init(
            project="flowmatching_MNIST",
            config={
                "setting": "class_conditional",
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim
            }
        )
    
    # Train Model
    train(class_cond_flow_matching,
          trainloader,
          optimizer,
          scheduler,
          device,
          args.epochs,
          target_dir,
          wandb_logging=(True if args.wandb_project else False))



    
    
    

