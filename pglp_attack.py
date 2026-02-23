"""
PGLP (Perception-Guided Large Perturbation) Attack Implementation
==================================================================

This script implements the PGLP-S and PGLP-C attacks described in the paper:
"Perception-Guided Large Perturbation Attacks against Adversarial Training"

The PGLP attacks use the PQP (Perceptual Quality Predictor) metric to generate
adversarial examples that maximize perturbation distance while maintaining
human-imperceptible quality.

Two attack variants are implemented:
1. PGLP-S: Search-based integration (PQP-guided gradient search)
2. PGLP-C: Change-based integration (two-phase refinement)

Requirements:
- PyTorch
- ART (Adversarial Robustness Toolbox)
- Pre-trained PQP model (see models/ directory)
- Pre-trained surrogate models for target dataset

Author: PGLP Research Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import feature computation module
from feature_computation import compute_features, get_dataset_config


class PGLPAttack:
    """
    Base class for PGLP attacks.
    
    Attributes:
    -----------
    pqp_model : object
        Pre-trained PQP (Perceptual Quality Predictor) model
    pqp_threshold : float
        Minimum acceptable PQP score (default: 0.7)
    device : torch.device
        Device for computations (CPU/GPU)
    """
    
    def __init__(self, pqp_model_path: str, pqp_threshold: float = 0.7, 
                 device: Optional[torch.device] = None,
                 dataset: str = 'cifar10'):
        """
        Initialize PGLP attack.
        
        Parameters:
        -----------
        pqp_model_path : str
            Path to pre-trained PQP model (.joblib file)
        pqp_threshold : float
            Minimum PQP score threshold (default: 0.7)
        device : torch.device, optional
            Computation device. If None, uses CUDA if available.
        """
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load PQP model
        print(f"Loading PQP model from: {pqp_model_path}")
        pqp_data = joblib.load(pqp_model_path)
        self.pqp_model = pqp_data['model']
        self.feature_names = pqp_data.get('feature_names', 
            ['SSIM', 'PSNR', 'UQI', 'VIF', 'LPIPS', 'AD_2', 'AD_inf', 'DTDB_2', 'DTDB_inf'])
        
        self.pqp_threshold = pqp_threshold
        print(f"PQP threshold set to: {pqp_threshold}")
        print(f"Using device: {self.device}")
        self.dataset = dataset
        print(f"Using dataset: {self.dataset}")
        
    def compute_pqp_score(self, original_img: np.ndarray, 
                         adversarial_img: np.ndarray) -> float:
        """
        Compute PQP score for an adversarial example.
        
        Parameters:
        -----------
        original_img : np.ndarray
            Original clean image
        adversarial_img : np.ndarray
            Adversarial perturbed image
            
        Returns:
        --------
        pqp_score : float
            PQP quality score (0-1, higher = better quality)
        """
        # Extract features (simplified version - full version needs sewar library)
        # In practice, compute: SSIM, PSNR, UQI, VIF, LPIPS, AD_2, AD_inf, DTDB_2, DTDB_inf
        
        # Placeholder for feature extraction
        # User should implement actual feature computation here
        features = self._extract_features(original_img, adversarial_img)
        
        # Predict PQP score
        pqp_score = self.pqp_model.predict(features.reshape(1, -1))[0]
        
        return float(pqp_score)
    
    def _extract_features(self, original: np.ndarray, 
                         adversarial: np.ndarray) -> np.ndarray:
        """
        Extract 9 features for PQP prediction using actual metric computation.
        
        Features computed: [SSIM, PSNR, UQI, VIF, LPIPS, AD_2, AD_inf, DTDB_2, DTDB_inf]
        Parameters:
        -----------
        original : np.ndarray
            Original image in [0, 1] range, shape (H, W, C) or (H, W)
        adversarial : np.ndarray
            Adversarial image in [0, 1] range, shape (H, W, C) or (H, W)
        Returns:
        --------
        features : np.ndarray
            9-dimensional feature vector
        """
        # Ensure images are in correct format (H, W, C) for feature computation
        if original.ndim == 3 and original.shape[0] == 3:
            # CHW -> HWC
            original = np.transpose(original, (1, 2, 0))
        if adversarial.ndim == 3 and adversarial.shape[0] == 3:
            # CHW -> HWC
            adversarial = np.transpose(adversarial, (1, 2, 0))
        
        # Clip to valid range
        original = np.clip(original, 0, 1).astype(np.float32)
        adversarial = np.clip(adversarial, 0, 1).astype(np.float32)
        
        # Compute features using the feature computation module
        # Note: DTDB uses a placeholder (0.5 * AD) since DBI image is not provided
        features = compute_features(original, adversarial, dataset=self.dataset)
        
        return features


class PGLP_S(PGLPAttack):
    """
    PGLP-S: Search-based integration attack.
    
    Integrates PQP into the loss function during gradient-based search.
    Uses penalty term to enforce perceptual quality constraint.
    
    Reference: Section IV-B of the paper
    """
    
    def __init__(self, surrogate_model: nn.Module, pqp_model_path: str,
                 pqp_threshold: float = 0.7, lambda_penalty: float = 1.0,
                 beta: float = 10.0, k: float = 10.0, **kwargs):
        """
        Initialize PGLP-S attack.
        
        Parameters:
        -----------
        surrogate_model : nn.Module
            Surrogate model for generating adversarial examples
        pqp_model_path : str
            Path to PQP model
        pqp_threshold : float
            Minimum PQP threshold
        lambda_penalty : float
            Weight for PQP penalty term (default: 1.0)
        beta : float
            Penalty magnitude parameter (default: 10.0)
        k : float
            Penalty steepness parameter (default: 10.0)
        """
        super().__init__(pqp_model_path, pqp_threshold, **kwargs)
        
        self.surrogate_model = surrogate_model.to(self.device)
        self.surrogate_model.eval()
        
        self.lambda_penalty = lambda_penalty
        self.beta = beta
        self.k = k
        
        print(f"PGLP-S initialized with lambda={lambda_penalty}, beta={beta}, k={k}")
    
    def compute_pqp_penalty(self, pqp_score: float) -> float:
        """
        Compute PQP penalty using sigmoid function.
        
        Penalty = -beta / (1 + exp(k * (pqp_score - threshold)))
        
        When pqp_score > threshold: penalty ≈ 0
        When pqp_score < threshold: penalty increases sharply
        """
        penalty = -self.beta / (1 + np.exp(self.k * (pqp_score - self.pqp_threshold)))
        return penalty
    
    def attack(self, image: torch.Tensor, true_label: int, 
               max_iterations: int = 100, step_size: float = 0.01,
               verbose: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial example using PGLP-S.
        
        Algorithm:
        1. Initialize x' = x
        2. For each iteration:
           a. Compute PQP penalty
           b. Compute total loss = L_class + lambda * P_penalty
           c. Update x' using gradient ascent
        3. Return when PQP < threshold or max iterations reached
        
        Parameters:
        -----------
        image : torch.Tensor
            Clean input image [C, H, W]
        true_label : int
            True class label
        max_iterations : int
            Maximum attack iterations
        step_size : float
            Step size for gradient updates
        verbose : bool
            Print progress information
            
        Returns:
        --------
        adv_image : torch.Tensor
            Generated adversarial example
        info : dict
            Attack information (success, iterations, PQP score, etc.)
        """
        image = image.to(self.device)
        original_image = image.clone()
        
        # Initialize adversarial image
        adv_image = image.clone().detach().requires_grad_(True)
        
        info = {
            'success': False,
            'iterations': 0,
            'final_pqp': 0.0,
            'perturbation_l2': 0.0,
            'perturbation_linf': 0.0
        }
        
        for iteration in range(max_iterations):
            if adv_image.grad is not None:
                adv_image.grad.zero_()
            
            # Forward pass through surrogate model
            output = self.surrogate_model(adv_image.unsqueeze(0))
            
            # Classification loss (we want to maximize loss = misclassify)
            loss_class = F.cross_entropy(output, 
                                        torch.tensor([true_label]).to(self.device))
            
            # Compute PQP score (detach for efficiency)
            with torch.no_grad():
                pqp_score = self.compute_pqp_score(
                    original_image.cpu().numpy(),
                    adv_image.detach().cpu().numpy()
                )
            
            # Compute PQP penalty
            penalty = self.compute_pqp_penalty(pqp_score)
            
            # Total loss: maximize classification loss + minimize penalty
            # Note: We negate loss_class because we want to maximize it
            total_loss = -loss_class + self.lambda_penalty * penalty
            
            # Backward pass
            total_loss.backward()
            
            # Update adversarial image
            with torch.no_grad():
                grad = adv_image.grad.sign()
                adv_image = adv_image + step_size * grad
                adv_image = torch.clamp(adv_image, 0, 1)
                adv_image = adv_image.detach().requires_grad_(True)
            
            # Check if adversarial and satisfies PQP constraint
            with torch.no_grad():
                pred = self.surrogate_model(adv_image.unsqueeze(0))
                predicted_label = pred.argmax().item()
                
                if predicted_label != true_label and pqp_score >= self.pqp_threshold:
                    perturbation = (adv_image - original_image).detach()
                    info.update({
                        'success': True,
                        'iterations': iteration + 1,
                        'final_pqp': pqp_score,
                        'perturbation_l2': torch.norm(perturbation, p=2).item(),
                        'perturbation_linf': torch.norm(perturbation, p=float('inf')).item()
                    })
                    
                    if verbose:
                        print(f"Success at iteration {iteration + 1}")
                        print(f"PQP score: {pqp_score:.4f}")
                        print(f"Perturbation L2: {info['perturbation_l2']:.4f}")
                    
                    return adv_image.detach(), info
                
                # Early termination if PQP drops too low
                if pqp_score < self.pqp_threshold - 0.1:
                    if verbose:
                        print(f"Early stop at iteration {iteration + 1}: PQP too low")
                    break
        
        # Attack failed or max iterations reached
        perturbation = (adv_image - original_image).detach()
        info.update({
            'iterations': max_iterations,
            'final_pqp': pqp_score,
            'perturbation_l2': torch.norm(perturbation, p=2).item(),
            'perturbation_linf': torch.norm(perturbation, p=float('inf')).item()
        })
        
        return adv_image.detach(), info


class PGLP_C(PGLPAttack):
    """
    PGLP-C: Change-based integration attack.
    
    Two-phase approach:
    1. Line search: Push AE away from decision boundary along DTDB direction
    2. Regional search: Trust region method to refine while maintaining PQP
    
    Reference: Section IV-B of the paper
    """
    
    def __init__(self, surrogate_model: nn.Module, pqp_model_path: str,
                 pqp_threshold: float = 0.7, **kwargs):
        """
        Initialize PGLP-C attack.
        
        Parameters:
        -----------
        surrogate_model : nn.Module
            Surrogate model
        pqp_model_path : str
            Path to PQP model
        pqp_threshold : float
            Minimum PQP threshold
        """
        super().__init__(pqp_model_path, pqp_threshold, **kwargs)
        self.surrogate_model = surrogate_model.to(self.device)
        self.surrogate_model.eval()
        
        print("PGLP-C initialized")
    
    def attack(self, original_image: torch.Tensor, initial_ae: torch.Tensor,
               true_label: int, max_line_search: int = 50,
               max_regional_search: int = 100, trust_region_radius: float = 0.1,
               verbose: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial example using PGLP-C.
        
        Algorithm:
        1. Start with initial AE x_0 (from DeepFool or other attack)
        2. Line Search Phase:
           a. Move along direction perpendicular to decision boundary
           b. Increase DTDB while PQP > threshold
        3. Regional Search Phase:
           a. Use trust region to explore locally
           b. Maximize distance while maintaining PQP
        
        Parameters:
        -----------
        original_image : torch.Tensor
            Original clean image
        initial_ae : torch.Tensor
            Initial adversarial example (e.g., from DeepFool)
        true_label : int
            True class label
        max_line_search : int
            Maximum line search iterations
        max_regional_search : int
            Maximum regional search iterations
        trust_region_radius : float
            Initial trust region radius
        verbose : bool
            Print progress
            
        Returns:
        --------
        adv_image : torch.Tensor
            Refined adversarial example
        info : dict
            Attack information
        """
        original_image = original_image.to(self.device)
        adv_image = initial_ae.clone().to(self.device)
        
        info = {
            'success': True,  # Assume success if initial AE is valid
            'line_search_iterations': 0,
            'regional_search_iterations': 0,
            'final_pqp': 0.0,
            'initial_distance': 0.0,
            'final_distance': 0.0
        }
        
        # Compute initial distance
        info['initial_distance'] = torch.norm(
            adv_image - original_image, p=2
        ).item()
        
        # ===== Phase 1: Line Search =====
        if verbose:
            print("Starting line search phase...")
        
        for iteration in range(max_line_search):
            # Compute PQP
            pqp_score = self.compute_pqp_score(
                original_image.cpu().numpy(),
                adv_image.detach().cpu().numpy()
            )
            
            if pqp_score < self.pqp_threshold:
                if verbose:
                    print(f"Line search stopped at iteration {iteration}: PQP = {pqp_score:.4f}")
                break
            
            # Move away from original along adversarial direction
            direction = (adv_image - original_image)
            direction = direction / (torch.norm(direction) + 1e-8)
            
            # Take step
            with torch.no_grad():
                adv_image = adv_image + 0.01 * direction
                adv_image = torch.clamp(adv_image, 0, 1)
            
            info['line_search_iterations'] = iteration + 1
        
        # ===== Phase 2: Regional Search =====
        if verbose:
            print("Starting regional search phase...")
        
        radius = trust_region_radius
        best_image = adv_image.clone()
        best_distance = torch.norm(best_image - original_image, p=2).item()
        
        for iteration in range(max_regional_search):
            # Sample points in trust region
            improved = False
            
            for _ in range(10):  # Sample 10 points
                noise = torch.randn_like(adv_image) * radius
                candidate = adv_image + noise
                candidate = torch.clamp(candidate, 0, 1)
                
                # Check PQP
                pqp_score = self.compute_pqp_score(
                    original_image.cpu().numpy(),
                    candidate.detach().cpu().numpy()
                )
                
                if pqp_score >= self.pqp_threshold:
                    distance = torch.norm(candidate - original_image, p=2).item()
                    
                    if distance > best_distance:
                        best_image = candidate.clone()
                        best_distance = distance
                        improved = True
            
            if improved:
                adv_image = best_image
                radius *= 1.5  # Expand trust region
            else:
                radius *= 0.5  # Shrink trust region
            
            info['regional_search_iterations'] = iteration + 1
            
            # Termination condition
            if radius < 0.001:
                if verbose:
                    print(f"Regional search converged at iteration {iteration}")
                break
        
        # Final PQP score
        final_pqp = self.compute_pqp_score(
            original_image.cpu().numpy(),
            adv_image.detach().cpu().numpy()
        )
        
        info.update({
            'final_pqp': final_pqp,
            'final_distance': torch.norm(adv_image - original_image, p=2).item()
        })
        
        if verbose:
            print(f"PGLP-C complete:")
            print(f"  Initial distance: {info['initial_distance']:.4f}")
            print(f"  Final distance: {info['final_distance']:.4f}")
            print(f"  Improvement: {(info['final_distance'] - info['initial_distance']):.4f}")
            print(f"  Final PQP: {final_pqp:.4f}")
        
        return adv_image.detach(), info


def demo_attack():
    """
    Demonstration of PGLP attacks on synthetic data.
    """
    print("="*70)
    print("PGLP Attack Demo")
    print("="*70)
    print("\nThis demo shows how to use PGLP-S and PGLP-C attacks.")
    print("For actual experiments, use pre-trained models and proper datasets.")
    print("="*70)
    
    # Create dummy model for demonstration
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3*32*32, 10)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = DummyModel()
    
    # Create dummy image
    image = torch.rand(3, 32, 32)
    label = 0
    
    print("\n[Demo 1] PGLP-S Attack")
    print("-" * 50)
    
    # Note: This will use placeholder PQP model
    # Replace with actual PQP model path for real experiments
    try:
        attack_s = PGLP_S(model, 'models/pqp_cifar10.joblib', pqp_threshold=0.7)
        adv_image_s, info_s = attack_s.attack(image, label, max_iterations=10, verbose=True)
        print(f"\nAttack result: {'Success' if info_s['success'] else 'Failed'}")
        print(f"Iterations: {info_s['iterations']}")
        print(f"Final PQP: {info_s['final_pqp']:.4f}")
    except Exception as e:
        print(f"Note: Demo requires actual PQP model. Error: {e}")
    
    print("\n[Demo 2] PGLP-C Attack")
    print("-" * 50)
    
    try:
        attack_c = PGLP_C(model, 'models/pqp_cifar10.joblib', pqp_threshold=0.7)
        initial_ae = image + torch.randn_like(image) * 0.1  # Dummy initial AE
        initial_ae = torch.clamp(initial_ae, 0, 1)
        
        adv_image_c, info_c = attack_c.attack(
            image, initial_ae, label, 
            max_line_search=10, max_regional_search=20, verbose=True
        )
        print(f"\nAttack complete:")
        print(f"Distance improvement: {(info_c['final_distance'] - info_c['initial_distance']):.4f}")
    except Exception as e:
        print(f"Note: Demo requires actual PQP model. Error: {e}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nFor actual usage:")
    print("1. Train/load surrogate models")
    print("2. Load pre-trained PQP model")
    print("3. Run PGLP-S or PGLP-C attacks")
    print("4. Evaluate on target models")
    print("="*70)


if __name__ == "__main__":
    demo_attack()
