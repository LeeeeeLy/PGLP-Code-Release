"""
PQP (Perceptual Quality Predictor) Training Demo
===============================================

This script demonstrates how to train the PQP metric using simulated human study data.
For the actual PQP models used in the paper, see the pre-trained models provided.

IMPORTANT NOTE ON IRB AND DATA PRIVACY:
---------------------------------------
Due to IRB regulations, we cannot release the raw human study data containing 
individual participant responses. However, the trained PQP models are released 
as they represent aggregated, anonymized predictions that cannot be reverse-engineered
to obtain raw human data.

The PQP metric is a Random Forest regression model that predicts human perceptual 
quality ratings based on image features. The model takes 9 input features and outputs 
a quality score between 0 and 1.

Author: PGLP Research Team
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_synthetic_human_data(n_samples=1000, random_state=42):
    """
    Generate synthetic training data simulating human perception ratings.
    
    In the actual study, this data comes from human participants rating 
    adversarial examples. Here we simulate the correlation patterns observed
    in our human study.
    
    Parameters:
    -----------
    n_samples : int
        Number of synthetic samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : np.ndarray, shape (n_samples, 9)
        Feature matrix with columns:
        [SSIM, PSNR, UQI, VIF, LPIPS, AD_2, AD_inf, DTDB_2, DTDB_inf]
    y : np.ndarray, shape (n_samples,)
        Human perception ratings (0-1 scale)
    """
    np.random.seed(random_state)
    
    # Generate features with realistic ranges
    # Traditional image quality metrics (higher = better quality)
    ssim = np.random.beta(2, 2, n_samples)  # Range [0, 1]
    psnr = np.random.gamma(5, 5, n_samples)  # Range typically [10, 50]
    uqi = np.random.beta(2, 2, n_samples)  # Range [0, 1]
    vif = np.random.beta(2, 3, n_samples)  # Range [0, 1]
    
    # Learned perceptual metric (lower = more similar)
    lpips = np.random.beta(2, 5, n_samples)  # Range [0, 1]
    
    # Distance metrics (lower = closer to original)
    ad_2 = np.random.gamma(2, 2, n_samples)  # Euclidean distance
    ad_inf = np.random.beta(3, 2, n_samples)  # Max perturbation
    dtdb_2 = np.random.gamma(2, 1.5, n_samples)  # Distance to decision boundary
    dtdb_inf = np.random.beta(3, 2, n_samples)  # Max DTDB
    
    X = np.column_stack([ssim, psnr, uqi, vif, lpips, ad_2, ad_inf, dtdb_2, dtdb_inf])
    
    # Generate synthetic human ratings based on observed correlations
    # Human ratings correlate positively with SSIM, PSNR, UQI, VIF
    # and negatively with LPIPS and distance metrics
    y = (
        0.35 * ssim +
        0.05 * (psnr / 50) +  # Normalize PSNR
        0.15 * uqi +
        0.10 * vif -
        0.25 * lpips -
        0.05 * (ad_2 / 10) -  # Normalize AD_2
        0.02 * ad_inf -
        0.02 * (dtdb_2 / 10) -
        0.01 * dtdb_inf +
        np.random.normal(0, 0.05, n_samples)  # Add noise
    )
    
    # Clip to valid range [0, 1]
    y = np.clip(y, 0, 1)
    
    return X, y


def train_pqp_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train a Random Forest model for PQP (Perceptual Quality Predictor).
    
    The model uses the same architecture as in the paper: Random Forest
    with hyperparameters optimized via cross-validation.
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and test feature matrices
    y_train, y_test : np.ndarray
        Training and test labels (human ratings)
    feature_names : list
        Names of the 9 features
        
    Returns:
    --------
    model : RandomForestRegressor
        Trained PQP model
    metrics : dict
        Dictionary containing model performance metrics
    """
    print("Training PQP model...")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Random Forest with hyperparameters from paper
    model = RandomForestRegressor(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Maximum tree depth
        min_samples_split=5,   # Minimum samples to split node
        min_samples_leaf=2,    # Minimum samples in leaf
        max_features='sqrt',   # Number of features to consider
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'correlation': correlation,
        'rmse': np.sqrt(mse)
    }
    
    print("\n" + "="*50)
    print("PQP Model Performance")
    print("="*50)
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Correlation with human ratings: {correlation:.4f}")
    print("="*50)
    
    # Feature importance analysis
    print("\nFeature Importance:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return model, metrics


def visualize_results(y_test, y_pred, dataset_name="Demo"):
    """
    Create visualization of PQP predictions vs actual human ratings.
    
    Parameters:
    -----------
    y_test : np.ndarray
        Actual human ratings
    y_pred : np.ndarray
        PQP predictions
    dataset_name : str
        Name of dataset for plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Human Rating', fontsize=12)
    axes[0].set_ylabel('PQP Predicted Rating', fontsize=12)
    axes[0].set_title(f'PQP Performance on {dataset_name}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('PQP Predicted Rating', fontsize=12)
    axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pqp_training_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to 'pqp_training_results.png'")
    plt.show()


def save_model(model, filepath, metadata=None):
    """
    Save trained PQP model to disk.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained PQP model
    filepath : str
        Path to save the model
    metadata : dict, optional
        Additional metadata to save with model
    """
    save_data = {
        'model': model,
        'feature_names': ['SSIM', 'PSNR', 'UQI', 'VIF', 'LPIPS', 
                         'AD_2', 'AD_inf', 'DTDB_2', 'DTDB_inf'],
        'metadata': metadata or {}
    }
    
    joblib.dump(save_data, filepath)
    print(f"\nModel saved to: {filepath}")
    print(f"File size: {np.round(os.path.getsize(filepath) / 1024, 2)} KB")


def main():
    """
    Main function demonstrating PQP training pipeline.
    """
    print("="*70)
    print("PQP (Perceptual Quality Predictor) Training Demo")
    print("="*70)
    print("\nThis script demonstrates how to train the PQP metric.")
    print("For actual models used in the paper, see pre-trained models.")
    print("="*70)
    
    # Step 1: Generate synthetic training data
    print("\n[Step 1/5] Generating synthetic training data...")
    X, y = generate_synthetic_human_data(n_samples=2000, random_state=42)
    
    feature_names = ['SSIM', 'PSNR', 'UQI', 'VIF', 'LPIPS', 
                     'AD_2', 'AD_inf', 'DTDB_2', 'DTDB_inf']
    
    print(f"Features: {feature_names}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Step 2: Split data
    print("\n[Step 2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Train PQP model
    print("\n[Step 3/5] Training Random Forest model...")
    model, metrics = train_pqp_model(X_train, y_train, X_test, y_test, feature_names)
    
    # Step 4: Visualize results
    print("\n[Step 4/5] Creating visualizations...")
    y_pred = model.predict(X_test)
    visualize_results(y_test, y_pred, dataset_name="Synthetic Demo Data")
    
    # Step 5: Save model
    print("\n[Step 5/5] Saving model...")
    metadata = {
        'description': 'Demo PQP model trained on synthetic data',
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test),
        'mse': metrics['mse'],
        'r2': metrics['r2'],
        'correlation': metrics['correlation'],
        'note': 'This is a demo model. Use pre-trained models for actual experiments.'
    }
    
    save_model(model, 'pqp_model_demo.joblib', metadata)
    
    # Save training data info
    training_info = pd.DataFrame({
        'Feature': feature_names,
        'Mean': np.mean(X, axis=0),
        'Std': np.std(X, axis=0),
        'Min': np.min(X, axis=0),
        'Max': np.max(X, axis=0)
    })
    training_info.to_csv('pqp_training_info.csv', index=False)
    print("Training info saved to: pqp_training_info.csv")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Load the model: joblib.load('pqp_model_demo.joblib')")
    print("2. Use model.predict(features) to get PQP scores")
    print("3. See test_pglp.py for usage examples")
    print("\nFor actual PGLP attacks, see pglp_attack.py")
    print("="*70)


if __name__ == "__main__":
    main()
