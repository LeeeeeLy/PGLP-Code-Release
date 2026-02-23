"""
Example: Running PGLP-C Attack
==============================

This script demonstrates how to use PGLP-C attack on CIFAR-10.

PGLP-C requires an initial adversarial example (from DeepFool or similar).
This example shows how to use PGLP-C with a simple initial AE.

Usage:
    python example_pglpc.py
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import os

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pglp_attack import PGLP_C


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


def create_initial_ae(model, image, label, device, epsilon=0.1):
    """
    Create a simple initial adversarial example using FGSM-like method.
    
    In practice, you would use DeepFool or another attack here.
    
    Parameters:
    -----------
    model : nn.Module
        Surrogate model
    image : torch.Tensor
        Original image
    label : int
        True label
    device : torch.device
        Device to use
    epsilon : float
        Perturbation magnitude
        
    Returns:
    --------
    initial_ae : torch.Tensor
        Initial adversarial example
    """
    image = image.clone().detach().requires_grad_(True)
    
    output = model(image.unsqueeze(0))
    loss = nn.CrossEntropyLoss()(output, torch.tensor([label]).to(device))
    loss.backward()
    
    # Create perturbation
    perturbation = image.grad.sign() * epsilon
    initial_ae = (image + perturbation).clamp(0, 1)
    
    return initial_ae


def main():
    print("="*60)
    print("PGLP-C Attack Example")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create surrogate model
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
    
    # Initialize PGLP-C attack
    print("\n[3] Initializing PGLP-C attack...")
    attack = PGLP_C(
        surrogate_model=surrogate_model,
        pqp_model_path=pqp_path,
        pqp_threshold=0.7,      # Minimum acceptable PQP score
        dataset='cifar10'      # Dataset name
    )
    
    # Load sample image
    print("\n[4] Preparing sample image...")
    image = torch.rand(3, 32, 32).to(device)
    true_label = 0  # Example class
    
    print(f"Image shape: {image.shape}")
    print(f"True label: {true_label}")
    
    # Create initial adversarial example
    print("\n[5] Creating initial adversarial example...")
    print("-" * 40)
    
    # Simple initialization (in practice, use DeepFool)
    initial_ae = image + torch.randn_like(image) * 0.05
    initial_ae = torch.clamp(initial_ae, 0, 1)
    
    initial_distance = torch.norm(initial_ae - image, p=2).item()
    print(f"Initial AE created")
    print(f"Initial L2 distance: {initial_distance:.4f}")
    
    # Run PGLP-C attack
    print("\n[6] Running PGLP-C attack...")
    print("-" * 40)
    
    adv_image, info = attack.attack(
        original_image=image,
        initial_ae=initial_ae,
        true_label=true_label,
        max_line_search=30,
        max_regional_search=50,
        trust_region_radius=0.1,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("Attack Results")
    print("="*60)
    print(f"Success: {info['success']}")
    print(f"Line search iterations: {info['line_search_iterations']}")
    print(f"Regional search iterations: {info['regional_search_iterations']}")
    print(f"Final PQP score: {info['final_pqp']:.4f}")
    print(f"Initial L2 distance: {info['initial_distance']:.4f}")
    print(f"Final L2 distance: {info['final_distance']:.4f}")
    print(f"Improvement: {info['final_distance'] - info['initial_distance']:.4f}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nTo use with your own models:")
    print("1. Replace SimpleCNN with your trained model")
    print("2. Use DeepFool or stronger attack for initial AE")
    print("3. Adjust parameters as needed")


if __name__ == "__main__":
    main()
