"""
Creates train and test dataloaders for MNIST dataset.
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 1

def create_dataloader(
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int=NUM_WORKERS
):
  
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader).
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """

  # Create MNIST datasets
  trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

  # Turn images into dataloaders
  train_dataloader = DataLoader(
      trainset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      testset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader