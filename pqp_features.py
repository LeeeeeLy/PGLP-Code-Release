"""
Dataset-Specific PQP Feature Configuration
==========================================

This module defines the feature subsets used by PQP models for each dataset.

According to the paper (Table 2), different datasets use different combinations
of the 9 features based on feature selection optimization.

Feature Mapping:
--------------
(1) SSIM  - Structural Similarity Index
(2) PSNR  - Peak Signal-to-Noise Ratio
(3) UQI   - Universal Quality Index
(4) VIF   - Visual Information Fidelity
(5) LPIPS - Learned Perceptual Image Patch Similarity
(6) AD_2  - Adversarial Distance (L2)
(7) AD_inf- Adversarial Distance (L∞)
(8) DTDB_2- Distance To Decision Boundary (L2)
(9) DTDB_inf- Distance To Decision Boundary (L∞)

Dataset-Specific Feature Subsets:
---------------------------------
- MNIST:      [1, 3, 8]     - SSIM, UQI, DTDB_2
- CIFAR-10:   [1, 3, 4, 5, 7, 8, 9] - SSIM, UQI, VIF, LPIPS, AD_inf, DTDB_2, DTDB_inf
- ImageNet:   [1, 2, 3, 4, 6, 7, 8] - SSIM, PSNR, UQI, VIF, AD_2, AD_inf, DTDB_2

Usage:
------
    from pqp_features import get_feature_subset, extract_features_for_dataset
    
    # Get feature indices for a dataset
    feature_indices = get_feature_subset('cifar10')
    print(f"CIFAR-10 uses features: {feature_indices}")
    
    # Extract and select features
    all_features = compute_all_features(original, adversarial, model)
    selected_features = extract_features_for_dataset(all_features, 'cifar10')
"""

import numpy as np
from typing import List, Dict, Tuple
import warnings

# Feature names for reference
FEATURE_NAMES = {
    1: 'SSIM',
    2: 'PSNR',
    3: 'UQI',
    4: 'VIF',
    5: 'LPIPS',
    6: 'AD_2',
    7: 'AD_inf',
    8: 'DTDB_2',
    9: 'DTDB_inf'
}

# Dataset-specific feature subsets (from Table 2 in paper)
DATASET_FEATURES = {
    'mnist': [1, 3, 8],  # SSIM, UQI, DTDB_2
    'cifar10': [1, 3, 4, 5, 7, 8, 9],  # SSIM, UQI, VIF, LPIPS, AD_inf, DTDB_2, DTDB_inf
    'cifar-10': [1, 3, 4, 5, 7, 8, 9],  # Alias
    'imagenet': [1, 2, 3, 4, 6, 7, 8],  # SSIM, PSNR, UQI, VIF, AD_2, AD_inf, DTDB_2
    'clustered_imagenet': [1, 2, 3, 4, 6, 7, 8],  # Same as ImageNet
    'clustered-imagenet': [1, 2, 3, 4, 6, 7, 8],  # Alias
}

# Model performance metrics (from Table 2)
DATASET_PERFORMANCE = {
    'mnist': {'mse': 0.009594, 'correlation': 0.963340, 'model': 'RF'},
    'cifar10': {'mse': 0.043119, 'correlation': 0.910328, 'model': 'RF'},
    'cifar-10': {'mse': 0.043119, 'correlation': 0.910328, 'model': 'RF'},
    'imagenet': {'mse': 0.035984, 'correlation': 0.912232, 'model': 'RF'},
    'clustered_imagenet': {'mse': 0.035984, 'correlation': 0.912232, 'model': 'RF'},
}


def get_feature_subset(dataset_name: str) -> List[int]:
    """
    Get the feature subset for a specific dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset ('mnist', 'cifar10', 'imagenet', etc.)
        
    Returns:
    --------
    feature_indices : List[int]
        List of feature indices (1-9) used by that dataset's PQP model
        
    Example:
    --------
    >>> get_feature_subset('mnist')
    [1, 3, 8]
    >>> get_feature_subset('cifar10')
    [1, 3, 4, 5, 7, 8, 9]
    """
    dataset_key = dataset_name.lower().replace('_', '-')
    
    if dataset_key not in DATASET_FEATURES:
        available = list(DATASET_FEATURES.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    return DATASET_FEATURES[dataset_key]


def get_feature_names(dataset_name: str) -> List[str]:
    """
    Get human-readable feature names for a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset
        
    Returns:
    --------
    feature_names : List[str]
        Names of features used by that dataset
        
    Example:
    --------
    >>> get_feature_names('mnist')
    ['SSIM', 'UQI', 'DTDB_2']
    """
    indices = get_feature_subset(dataset_name)
    return [FEATURE_NAMES[i] for i in indices]


def extract_features_for_dataset(all_features: np.ndarray, 
                                 dataset_name: str) -> np.ndarray:
    """
    Extract the relevant feature subset for a specific dataset.
    
    This function takes all 9 computed features and returns only the
    subset used by the specified dataset's PQP model.
    
    Parameters:
    -----------
    all_features : np.ndarray, shape (..., 9)
        Array containing all 9 features in order:
        [SSIM, PSNR, UQI, VIF, LPIPS, AD_2, AD_inf, DTDB_2, DTDB_inf]
    dataset_name : str
        Name of dataset
        
    Returns:
    --------
    selected_features : np.ndarray
        Array containing only the features used by that dataset
        
    Example:
    --------
    >>> all_features = np.array([0.8, 25.0, 0.9, 0.7, 0.3, 5.0, 0.1, 3.0, 0.05])
    >>> extract_features_for_dataset(all_features, 'mnist')
    array([0.8, 0.9, 3.0])  # SSIM, UQI, DTDB_2
    """
    # Convert 1-based indices to 0-based
    indices_0based = [i - 1 for i in get_feature_subset(dataset_name)]
    
    if all_features.ndim == 1:
        return all_features[indices_0based]
    else:
        return all_features[..., indices_0based]


def validate_feature_computation(features_dict: Dict[str, float]) -> bool:
    """
    Validate that all 9 features are present and in valid ranges.
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary with feature names as keys and values
        
    Returns:
    --------
    valid : bool
        True if all features are present and valid
    """
    required_features = list(FEATURE_NAMES.values())
    
    # Check all features present
    for feat in required_features:
        if feat not in features_dict:
            warnings.warn(f"Missing feature: {feat}")
            return False
    
    # Check value ranges (approximate)
    ranges = {
        'SSIM': (0, 1),
        'PSNR': (0, 100),
        'UQI': (0, 1),
        'VIF': (0, 1),
        'LPIPS': (0, 1),
        'AD_2': (0, 100),
        'AD_inf': (0, 1),
        'DTDB_2': (0, 100),
        'DTDB_inf': (0, 1)
    }
    
    for feat, (min_val, max_val) in ranges.items():
        val = features_dict[feat]
        if not (min_val <= val <= max_val):
            warnings.warn(f"Feature {feat} value {val} out of range [{min_val}, {max_val}]")
            return False
    
    return True


def print_dataset_info(dataset_name: str):
    """
    Print information about a dataset's PQP configuration.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset
    """
    dataset_key = dataset_name.lower().replace('_', '-')
    
    if dataset_key not in DATASET_FEATURES:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    features = get_feature_subset(dataset_name)
    feature_names = get_feature_names(dataset_name)
    performance = DATASET_PERFORMANCE.get(dataset_key, {})
    
    print(f"\n{'='*60}")
    print(f"PQP Configuration for {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"\nFeatures used ({len(features)}/9):")
    for idx, name in zip(features, feature_names):
        print(f"  ({idx}) {name}")
    
    print(f"\nFeatures NOT used:")
    unused = [i for i in range(1, 10) if i not in features]
    for idx in unused:
        print(f"  ({idx}) {FEATURE_NAMES[idx]}")
    
    if performance:
        print(f"\nModel Performance:")
        print(f"  Model type: {performance.get('model', 'Unknown')}")
        print(f"  MSE: {performance.get('mse', 'Unknown')}")
        print(f"  Correlation: {performance.get('correlation', 'Unknown')}")
    
    print(f"{'='*60}\n")


def get_feature_importance_info():
    """
    Get information about feature importance across datasets.
    
    Returns:
    --------
    importance_info : dict
        Dictionary showing which features are used by which datasets
    """
    info = {}
    
    for feat_num, feat_name in FEATURE_NAMES.items():
        datasets_using = []
        for dataset, features in DATASET_FEATURES.items():
            if feat_num in features:
                datasets_using.append(dataset)
        
        info[feat_name] = {
            'feature_number': feat_num,
            'datasets': datasets_using,
            'usage_count': len(datasets_using)
        }
    
    return info


# Example usage
if __name__ == "__main__":
    print("PQP Feature Configuration")
    print("="*60)
    
    # Show info for each dataset
    for dataset in ['mnist', 'cifar10', 'imagenet']:
        print_dataset_info(dataset)
    
    # Show feature usage across datasets
    print("\nFeature Usage Across Datasets:")
    print("-"*60)
    importance = get_feature_importance_info()
    
    for feat_name in ['SSIM', 'PSNR', 'UQI', 'VIF', 'LPIPS', 
                      'AD_2', 'AD_inf', 'DTDB_2', 'DTDB_inf']:
        info = importance[feat_name]
        datasets = ', '.join(info['datasets'])
        print(f"{feat_name:12s} - used by {info['usage_count']}/3 datasets: {datasets}")
    
    # Example: Extract features
    print("\n" + "="*60)
    print("Example: Feature Extraction")
    print("="*60)
    
    # Simulate computed features
    all_features = np.array([
        0.85,   # (1) SSIM
        28.5,   # (2) PSNR
        0.92,   # (3) UQI
        0.78,   # (4) VIF
        0.25,   # (5) LPIPS
        4.5,    # (6) AD_2
        0.08,   # (7) AD_inf
        3.2,    # (8) DTDB_2
        0.05    # (9) DTDB_inf
    ])
    
    print(f"\nAll 9 features: {all_features}")
    
    for dataset in ['mnist', 'cifar10', 'imagenet']:
        selected = extract_features_for_dataset(all_features, dataset)
        feature_names = get_feature_names(dataset)
        print(f"\n{dataset.upper()}:")
        print(f"  Selected features: {selected}")
        print(f"  Feature names: {feature_names}")
