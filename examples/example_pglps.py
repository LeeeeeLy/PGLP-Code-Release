"""
Example: Running PGLP-S Attack
==============================

This script demonstrates how to use PGLP-S attack on CIFAR-10.

Usage:
    python example_pglps.py
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pglp_attack import PGLP_S


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 demo."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("="*60)
    print("PGLP-S Attack Example")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create surrogate model (replace with your trained model)
    print("\n[1] Loading surrogate model...")
    surrogate_model = SimpleCNN()
    surrogate_model.to(device)
    surrogate_model.eval()
    print("Surrogate model created (SimpleCNN)")
    
    # Load PQP model
    print("\n[2] Loading PQP model...")
    # Path to parent directory for models
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pqp_path = os.path.join(parent_dir, 'models', 'pqp_cifar10.joblib')
    
    if not os.path.exists(pqp_path):
        print(f"ERROR: PQP model not found at {pqp_path}")
        print("Please ensure PQP models are in the models/ directory")
        return
    
    # Initialize PGLP-S attack
    print("\n[3] Initializing PGLP-S attack...")
    attack = PGLP_S(
        surrogate_model=surrogate_model,
        pqp_model_path=pqp_path,
        pqp_threshold=0.7,      # Minimum acceptable PQP score
        lambda_penalty=1.0,    # Weight for PQP penalty
        beta=10.0,             # Penalty steepness
        k=10.0,                # Penalty scaling
        dataset='cifar10'      # Dataset name
    )
    
    # Load sample image (CIFAR-10)
    print("\n[4] Preparing sample image...")
    # Use image for demo ( randomreplace with actual CIFAR-10 image)
    image = torch.rand(3, 32, 32).to(device)
    true_label = 0  # Example: 'airplane' class
    
    print(f"Image shape: {image.shape}")
    print(f"True label: {true_label}")
    
    # Run attack
    print("\n[5] Running PGLP-S attack...")
    print("-" * 40)
    
    adv_image, info = attack.attack(
        image=image,
        true_label=true_label,
        max_iterations=50,
        step_size=0.01,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("Attack Results")
    print("="*60)
    print(f"Success: {info['success']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final PQP score: {info['final_pqp']:.4f}")
    print(f"Perturbation L2: {info['perturbation_l2']:.4f}")
    print(f"Perturbation L-inf: {info['perturbation_linf']:.4f}")
    
    # Compute actual distances
    perturbation = (adv_image - image).detach()
    actual_l2 = torch.norm(perturbation, p=2).item()
    actual_linf = torch.norm(perturbation, p=float('inf')).item()
    
    print(f"\nActual L2 distance: {actual_l2:.4f}")
    print(f"Actual L-inf distance: {actual_linf:.4f}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nTo use with your own models:")
    print("1. Replace SimpleCNN with your trained model")
    print("2. Load actual CIFAR-10 images")
    print("3. Adjust pqp_threshold as needed")


if __name__ == "__main__":
    main()
