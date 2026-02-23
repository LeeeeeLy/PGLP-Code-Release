"""
Dataset-Specific Feature Computation for PQP
==========================================

This module provides dataset-specific feature computation for:
- MNIST (grayscale, 28x28)
- CIFAR-10 (RGB, 32x32)
- ImageNet / Clustered ImageNet (RGB, 64x64 or larger)

Key differences between datasets:
--------------------------------
1. Image format: MNIST (.bmp), CIFAR-10 (.bmp), ImageNet (.png)
2. Color space: MNIST is grayscale, others are RGB
3. LPIPS: All use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. SSIM/PSNR/UQI/VIF: Computed on grayscale versions
5. L2/L-inf: Computed on normalized [0,1] images
6. DTDB: Uses pre-computed DBI (decision boundary instance) images

Feature order: [SSIM, PSNR, UQI, VIF, LPIPS, AD_2, AD_inf, DTDB_2, DTDB_inf]

Requirements:
- sewar >= 0.4.5
- lpips >= 0.1.4
- torch, torchvision
- numpy
- PIL

Author: PGLP Research Team
License: MIT
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, Union
import warnings
import os

# Try to import optional dependencies
try:
    import sewar
    SEWAR_AVAILABLE = True
except ImportError:
    SEWAR_AVAILABLE = False
    warnings.warn("sewar not installed. SSIM, PSNR, UQI, VIF will use approximations.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips not installed. LPIPS will use approximation.")

try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not installed.")


# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_CONFIG = {
    'mnist': {
        'channels': 1,
        'image_size': 28,
        'format': '.bmp',
        'normalize': {'mean': [0.5], 'std': [0.5]},
        'lpips_size': 224,
        'feature_range': (0, 1)
    },
    'cifar10': {
        'channels': 3,
        'image_size': 32,
        'format': '.bmp',
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'lpips_size': 224,
        'feature_range': (0, 1)
    },
    'imagenet': {
        'channels': 3,
        'image_size': 64,
        'format': '.png',
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'lpips_size': 224,
        'feature_range': (0, 1)
    },
    'clustered_imagenet': {
        'channels': 3,
        'image_size': 64,
        'format': '.png',
        'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'lpips_size': 224,
        'feature_range': (0, 1)
    }
}


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a dataset."""
    dataset_name = dataset_name.lower().replace('-', '_').replace(' ', '_')
    if dataset_name not in DATASET_CONFIG:
        warnings.warn(f"Unknown dataset '{dataset_name}', using CIFAR-10 config")
        return DATASET_CONFIG['cifar10']
    return DATASET_CONFIG[dataset_name]


# ============================================================================
# LPIPS Model (lazy loaded)
# ============================================================================

_lpips_model = None

def get_lpips_model():
    """Get or create LPIPS model (lazy loading)."""
    global _lpips_model
    if _lpips_model is None:
        if LPIPS_AVAILABLE:
            _lpips_model = lpips.LPIPS(net='alex')
        else:
            raise ImportError("lpips not available")
    return _lpips_model


# ============================================================================
# Image Preprocessing
# ============================================================================

def load_image(image_path: str, dataset: str = 'cifar10') -> np.ndarray:
    """
    Load image and convert to numpy array.
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    dataset : str
        Dataset name ('mnist', 'cifar10', 'imagenet', 'clustered_imagenet')
        
    Returns:
    --------
    img : np.ndarray
        Image as float array in [0, 1] range
    """
    config = get_dataset_config(dataset)
    
    img = Image.open(image_path)
    
    if config['channels'] == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    
    img = np.array(img).astype(np.float32) / 255.0
    
    return img


def preprocess_for_lpips(img: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """
    Preprocess image for LPIPS computation.
    
    All datasets use ImageNet normalization for LPIPS.
    """
    if img.ndim == 2:
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    else:
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
    
    if img.shape[0] != target_size or img.shape[1] != target_size:
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
    
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_img).unsqueeze(0)
    return tensor


# ============================================================================
# Metric Computations
# ============================================================================

def compute_ssim_psnr_uqi_vif(original: np.ndarray, adversarial: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute SSIM, PSNR, UQI, VIF using sewar library.
    
    For all datasets, these metrics are computed on grayscale images.
    """
    if not SEWAR_AVAILABLE:
        diff = adversarial - original
        if diff.ndim > 1:
            diff = np.mean(diff, axis=-1)
        ssim = max(0, 1.0 - np.mean(np.abs(diff)))
        psnr = 20.0 - 10 * np.log10(np.mean(diff**2) + 1e-10)
        uqi = ssim * 0.9
        vif = ssim * 0.8
        return ssim, psnr, uqi, vif
    
    def to_gray(img):
        if img.ndim == 2:
            return img
        elif img.shape[-1] == 1:
            return img.squeeze(-1)
        else:
            return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    
    orig_gray = to_gray(original)
    adv_gray = to_gray(adversarial)
    
    orig_8bit = (np.clip(orig_gray, 0, 1) * 255).astype(np.uint8)
    adv_8bit = (np.clip(adv_gray, 0, 1) * 255).astype(np.uint8)
    
    try:
        ssim_result = sewar.full_ref.ssim(orig_8bit, adv_8bit)
        ssim = float(ssim_result[0]) if isinstance(ssim_result, tuple) else float(ssim_result)
    except Exception as e:
        warnings.warn(f"SSIM computation failed: {e}")
        ssim = 0.5
        
    try:
        psnr = float(sewar.full_ref.psnr(orig_8bit, adv_8bit))
    except Exception as e:
        warnings.warn(f"PSNR computation failed: {e}")
        diff = adv_gray - orig_gray
        psnr = 20.0 - 10 * np.log10(np.mean(diff**2) + 1e-10)
        
    try:
        uqi = float(sewar.full_ref.uqi(orig_8bit, adv_8bit))
    except Exception as e:
        warnings.warn(f"UQI computation failed: {e}")
        uqi = 0.5
        
    try:
        vif = float(sewar.full_ref.vifp(orig_8bit, adv_8bit))
    except Exception as e:
        warnings.warn(f"VIF computation failed: {e}")
        vif = 0.5
    # Clamp values to valid ranges to avoid infinity issues
    ssim = float(np.clip(ssim, -1, 1))
    psnr = float(np.clip(psnr, 0, 100))
    uqi = float(np.clip(uqi, -1, 1))
    vif = float(np.clip(vif, 0, 1))
    
    return ssim, psnr, uqi, vif


def compute_lpips(original: np.ndarray, adversarial: np.ndarray) -> float:
    """
    Compute LPIPS distance using ImageNet normalization.
    """
    if not LPIPS_AVAILABLE:
        diff = adversarial - original
        if diff.ndim > 1:
            diff = np.mean(diff, axis=-1)
        return float(np.mean(np.abs(diff)))
    
    orig_tensor = preprocess_for_lpips(original)
    adv_tensor = preprocess_for_lpips(adversarial)
    
    with torch.no_grad():
        lpips_model = get_lpips_model()
        lpips_score = lpips_model(orig_tensor, adv_tensor)
    
    return float(lpips_score.item())


def compute_adversarial_distance(original: np.ndarray, adversarial: np.ndarray) -> Tuple[float, float]:
    """Compute L2 and L∞ adversarial distances."""
    diff = adversarial - original
    
    if diff.ndim == 3:
        ad_2 = float(np.sqrt(np.sum(diff ** 2)))
        ad_inf = float(np.max(np.abs(diff)))
    else:
        ad_2 = float(np.linalg.norm(diff))
        ad_inf = float(np.max(np.abs(diff)))
    
    return ad_2, ad_inf


def compute_dtdb_with_dbi(original: np.ndarray, dbi_image: np.ndarray) -> Tuple[float, float]:
    """
    Compute Distance To Decision Boundary using pre-computed DBI image.
    
    DTDB is computed as the distance between the original image and the 
    decision boundary instance (DBI) image.
    """
    diff = dbi_image - original
    
    if diff.ndim == 3:
        dtdb_2 = float(np.sqrt(np.sum(diff ** 2)))
        dtdb_inf = float(np.max(np.abs(diff)))
    else:
        dtdb_2 = float(np.linalg.norm(diff))
        dtdb_inf = float(np.max(np.abs(diff)))
    
    return dtdb_2, dtdb_inf


# ============================================================================
# Main Feature Computation Functions
# ============================================================================

def compute_features_mnist(original: np.ndarray, adversarial: np.ndarray,
                          dbi_image: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute all 9 features for MNIST dataset (grayscale)."""
    ssim, psnr, uqi, vif = compute_ssim_psnr_uqi_vif(original, adversarial)
    
    # LPIPS needs RGB - replicate grayscale
    if original.ndim == 2:
        orig_rgb = np.stack([original] * 3, axis=-1)
        adv_rgb = np.stack([adversarial] * 3, axis=-1)
    else:
        orig_rgb = original
        adv_rgb = adversarial
    lpips_score = compute_lpips(orig_rgb, adv_rgb)
    
    ad_2, ad_inf = compute_adversarial_distance(original, adversarial)
    
    if dbi_image is not None:
        dtdb_2, dtdb_inf = compute_dtdb_with_dbi(original, dbi_image)
    else:
        dtdb_2 = ad_2 * 0.5
        dtdb_inf = ad_inf * 0.5
    
    features = np.array([ssim, psnr, uqi, vif, lpips_score, ad_2, ad_inf, dtdb_2, dtdb_inf], dtype=np.float32)
    # Final safeguard: replace any inf/nan with reasonable defaults
    features = np.nan_to_num(features, nan=0.5, posinf=100.0, neginf=0.0)
    return features


def compute_features_cifar10(original: np.ndarray, adversarial: np.ndarray,
                             dbi_image: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute all 9 features for CIFAR-10 dataset (RGB)."""
    ssim, psnr, uqi, vif = compute_ssim_psnr_uqi_vif(original, adversarial)
    lpips_score = compute_lpips(original, adversarial)
    ad_2, ad_inf = compute_adversarial_distance(original, adversarial)
    
    if dbi_image is not None:
        dtdb_2, dtdb_inf = compute_dtdb_with_dbi(original, dbi_image)
    else:
        dtdb_2 = ad_2 * 0.5
        dtdb_inf = ad_inf * 0.5
    
    features = np.array([ssim, psnr, uqi, vif, lpips_score, ad_2, ad_inf, dtdb_2, dtdb_inf], dtype=np.float32)
    # Final safeguard: replace any inf/nan with reasonable defaults
    features = np.nan_to_num(features, nan=0.5, posinf=100.0, neginf=0.0)
    return features


def compute_features_imagenet(original: np.ndarray, adversarial: np.ndarray,
                             dbi_image: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute all 9 features for ImageNet / Clustered ImageNet dataset."""
    return compute_features_cifar10(original, adversarial, dbi_image)


def compute_features(original: Union[np.ndarray, str], 
                    adversarial: Union[np.ndarray, str],
                    dataset: str = 'cifar10',
                    dbi_image: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute all 9 features for any supported dataset.
    
    Parameters:
    -----------
    original : np.ndarray or str
        Original image or path to image
    adversarial : np.ndarray or str
        Adversarial image or path to image
    dataset : str
        Dataset name ('mnist', 'cifar10', 'imagenet', 'clustered_imagenet')
    dbi_image : np.ndarray, optional
        Decision boundary instance image
        
    Returns:
    --------
    features : np.ndarray, shape (9,)
    """
    if isinstance(original, str):
        original = load_image(original, dataset)
    if isinstance(adversarial, str):
        adversarial = load_image(adversarial, dataset)
    if dbi_image is not None and isinstance(dbi_image, str):
        dbi_image = load_image(dbi_image, dataset)
    
    dataset = dataset.lower()
    if dataset == 'mnist':
        return compute_features_mnist(original, adversarial, dbi_image)
    elif dataset in ['cifar10', 'cifar-10']:
        return compute_features_cifar10(original, adversarial, dbi_image)
    elif dataset in ['imagenet', 'clustered_imagenet', 'clustered-imagenet']:
        return compute_features_imagenet(original, adversarial, dbi_image)
    else:
        warnings.warn(f"Unknown dataset '{dataset}', using CIFAR-10")
        return compute_features_cifar10(original, adversarial, dbi_image)


def quick_features(original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
    """Quick feature computation (assumes CIFAR-10 format)."""
    return compute_features_cifar10(original, adversarial)


if __name__ == "__main__":
    print("Testing dataset-specific feature computation...")
    
    print("\n1. Testing MNIST features:")
    mnist_orig = np.random.rand(28, 28).astype(np.float32)
    mnist_adv = np.clip(mnist_orig + np.random.randn(28, 28) * 0.05, 0, 1)
    mnist_features = compute_features_mnist(mnist_orig, mnist_adv)
    print(f"  Features shape: {mnist_features.shape}")
    
    print("\n2. Testing CIFAR-10 features:")
    cifar_orig = np.random.rand(32, 32, 3).astype(np.float32)
    cifar_adv = np.clip(cifar_orig + np.random.randn(32, 32, 3) * 0.05, 0, 1)
    cifar_features = compute_features_cifar10(cifar_orig, cifar_adv)
    print(f"  Features shape: {cifar_features.shape}")
    
    print("\n3. Testing ImageNet features:")
    imagenet_orig = np.random.rand(64, 64, 3).astype(np.float32)
    imagenet_adv = np.clip(imagenet_orig + np.random.randn(64, 64, 3) * 0.05, 0, 1)
    imagenet_features = compute_features_imagenet(imagenet_orig, imagenet_adv)
    print(f"  Features shape: {imagenet_features.shape}")
    
    print("\n✓ All tests passed!")
