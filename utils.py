#!/usr/bin/env python

import torch
from timeit import default_timer as timer

def set_seeds(seed: int=42):
    """
    Set the seed for reproducibility.
    
    Args:
        seed (int): The seed value to use. DEFAULT = 42
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set device agnostic code

def device_setup():
    """
    Set up device-agnostic configurations
    """
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if device is cuda, then clear the cache first
    if device == "cuda":
        torch.cuda.empty_cache()
        
    #print(f"Using device: {device}")
    
    return device

### 0.3 Set training timer

def training_time(start: float,
                  end: float,
                  device: torch.device) -> None:
    """Prints difference between training start and end
    """
    total_time = end - start
    print(f"Total time on {device}: {total_time: .3f} seconds")
    #return total_time

# train-test split

# def train_test_split(X, split=0.8):
#     # Split training and test datasets

#     N = X.shape[0]

#     split = int(N * split)
#     X_train = X[:split]
#     X_test = X[split:]

#     return X_train, X_test
