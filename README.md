# PGLP Attack Implementation - Code Release

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the PGLP (Perception-Guided Large Perturbation) attacks from the paper:

**"Perception-Guided Large Perturbation Attacks against Adversarial Training of Neural Networks"**  


## Overview

PGLP attacks exploit the gap between mathematical distance metrics (e.g., Lp norms) and human perceptual quality to generate adversarial examples that:
- Maximize perturbation distance beyond adversarial training thresholds
- Maintain human-imperceptible quality measured by PQP (Perceptual Quality Predictor)
- Achieve high attack success rates against state-of-the-art defenses

### Two Attack Variants

1. **PGLP-S** (Search-based): Integrates PQP into gradient-based search with penalty term
2. **PGLP-C** (Change-based): Two-phase refinement starting from initial adversarial examples

## Repository Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pglp_attack.py              # Main PGLP attack implementations
├── train_pqp_demo.py           # PQP training demo with synthetic data
├── test_pglp.py                # Test suite
├── models/                     # Pre-trained PQP models
│   ├── pqp_cifar10.joblib     # PQP for CIFAR-10
│   ├── pqp_mnist.joblib       # PQP for MNIST
│   └── pqp_imagenet.joblib    # PQP for ImageNet
└── examples/                   # Usage examples
    ├── example_pglps.py
    └── example_pglpc.py
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- scikit-learn
- ART (Adversarial Robustness Toolbox)
- Additional dependencies in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/LeeeeeLy/PGLP-Code-Release.git
cd PGLP-Code-Release

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
##  What This Code Does

1. **PGLP-S Attack** (`pglp_attack.py` - Class `PGLP_S`)
   - Search-based gradient attack with PQP penalty
   - Integrates perceptual quality constraint into loss function
   - Suitable for generating adversarial examples from scratch

2. **PGLP-C Attack** (`pglp_attack.py` - Class `PGLP_C`)
   - Two-phase refinement attack
   - Phase 1: Line search to push away from decision boundary
   - Phase 2: Trust region method for local refinement
   - Requires initial adversarial example (e.g., from DeepFool)

3. **PQP Training Demo** (`train_pqp_demo.py`)
   - Demonstrates how PQP models are trained
   - Uses synthetic data (actual human study data protected by IRB)
   - Shows the complete training pipeline

4. **Test Suite** (`test_pglp.py`)
   - 6 comprehensive tests
   - Verifies all components work correctly
   - Can be run with: `python test_pglp.py`

##  IRB & Data Privacy

**IMPORTANT**: Due to IRB regulations, we CANNOT release:
- ❌ Raw human study participant responses
- ❌ Individual rating data
- ❌ Survey platform raw exports

**We CAN and DO release**:
- ✅ Pre-trained PQP models (aggregated predictions)
- ✅ Complete attack implementation
- ✅ Training methodology and demo code
- ✅ Survey protocol documentation

## Quick Start

### 1. Test the Installation

```bash
python test_pglp.py
```

Expected output: All 6 tests should pass.

### 2. Load Pre-trained PQP Model

```python
import joblib

# Load PQP model for CIFAR-10
pqp_data = joblib.load('models/pqp_cifar10.joblib')
pqp_model = pqp_data['model']
feature_names = pqp_data['feature_names']

print(f"PQP model loaded: {pqp_model}")
print(f"Features: {feature_names}")
```

### 3. Run PGLP-S Attack

```python
import torch
from pglp_attack import PGLP_S
from your_models import YourModel

# Load surrogate model
surrogate_model = YourModel()
surrogate_model.load_state_dict(torch.load('path/to/model.pt'))

# Initialize PGLP-S attack
attack = PGLP_S(
    surrogate_model=surrogate_model,
    pqp_model_path='models/pqp_cifar10.joblib',
    pqp_threshold=0.7,
    lambda_penalty=1.0
)

# Generate adversarial example
clean_image = torch.rand(3, 32, 32)  # Your input image
true_label = 0  # True class label

adv_image, info = attack.attack(
    image=clean_image,
    true_label=true_label,
    max_iterations=100,
    step_size=0.01,
    verbose=True
)

print(f"Attack success: {info['success']}")
print(f"PQP score: {info['final_pqp']:.4f}")
print(f"Perturbation L2: {info['perturbation_l2']:.4f}")
```

### 4. Run PGLP-C Attack

```python
from pglp_attack import PGLP_C
from art.attacks.evasion import DeepFool

# Initialize PGLP-C
attack_c = PGLP_C(
    surrogate_model=surrogate_model,
    pqp_model_path='models/pqp_cifar10.joblib',
    pqp_threshold=0.7
)

# First generate initial AE using DeepFool
deepfool = DeepFool(surrogate_model)
initial_ae = deepfool.generate(x=clean_image.numpy())
initial_ae = torch.from_numpy(initial_ae).float()

# Refine with PGLP-C
adv_image_c, info_c = attack_c.attack(
    original_image=clean_image,
    initial_ae=initial_ae,
    true_label=true_label,
    max_line_search=50,
    max_regional_search=100,
    verbose=True
)

print(f"Distance improved from {info_c['initial_distance']:.4f} to {info_c['final_distance']:.4f}")
```

## Training PQP from Scratch

While we provide pre-trained PQP models, you can also train your own:

### Important Note on Data Privacy

**Due to IRB regulations, we cannot release raw human study data containing individual participant responses.** However, we provide:

1. **Pre-trained PQP models**: These can be released as they represent aggregated, anonymized predictions
2. **Training demo script**: `train_pqp_demo.py` shows the training process using synthetic data
3. **Protocol documentation**: Detailed methodology for conducting human studies

### Training Process

1. **Generate Adversarial Examples**: Create diverse AEs with varying perturbation magnitudes
2. **Human Study**: Collect human perceptual ratings (IRB approval required)
3. **Feature Extraction**: Compute 9 features (SSIM, PSNR, UQI, VIF, LPIPS, AD_2, AD_inf, DTDB_2, DTDB_inf)
4. **Model Training**: Train Random Forest regressor

```python
# See train_pqp_demo.py for complete example
from train_pqp_demo import generate_synthetic_human_data, train_pqp_model

# Generate training data (replace with actual human ratings)
X, y = generate_synthetic_human_data(n_samples=1000)

# Train PQP model
model, metrics = train_pqp_model(X_train, y_train, X_test, y_test, feature_names)

# Save model
joblib.dump({'model': model, 'feature_names': feature_names}, 'my_pqp_model.joblib')
```

## Computing PQP Features

The PQP model requires 9 input features:

| Feature | Description | Range | Library |
|---------|-------------|-------|---------|
| SSIM | Structural Similarity | [0, 1] | `sewar` |
| PSNR | Peak Signal-to-Noise Ratio | [0, ∞] | `sewar` |
| UQI | Universal Quality Index | [0, 1] | `sewar` |
| VIF | Visual Information Fidelity | [0, 1] | `sewar` |
| LPIPS | Learned Perceptual Similarity | [0, 1] | `lpips` |
| AD_2 | L2 adversarial distance | [0, ∞] | numpy |
| AD_inf | L∞ adversarial distance | [0, 1] | numpy |
| DTDB_2 | L2 distance to decision boundary | [0, ∞] | Custom |
| DTDB_inf | L∞ distance to decision boundary | [0, 1] | Custom |

### Example Feature Computation

```python
import sewar
import numpy as np

import lpips  # pip install lpips

def compute_features(original, adversarial, model):
    # Convert to 8-bit for sewar
    orig_8bit = (original * 255).astype(np.uint8)
    adv_8bit = (adversarial * 255).astype(np.uint8)
    
    # Image quality metrics (using sewar library)
    ssim = sewar.full_ref.ssim(orig_8bit, adv_8bit)[0]
    psnr = sewar.full_ref.psnr(orig_8bit, adv_8bit)
    uqi = sewar.full_ref.uqi(orig_8bit, adv_8bit)
    vif = sewar.full_ref.vifp(orig_8bit, adv_8bit)
    
    # Learned perceptual metric (LPIPS)
    # Note: LPIPS requires PyTorch tensors and a pre-trained network
    loss_fn = lpips.LPIPS(net='alex')  # Use AlexNet backbone
    orig_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0)
    adv_tensor = torch.from_numpy(adversarial).permute(2, 0, 1).unsqueeze(0)
    lpips_score = loss_fn(orig_tensor, adv_tensor).item()
    
    # Distance metrics
    diff = adversarial - original
    ad_2 = np.linalg.norm(diff)
    ad_inf = np.max(np.abs(diff))
    
    # DTDB (Distance To Decision Boundary) - requires decision boundary estimation
    # See paper Section IV-A for DTDB computation method
    dtdb_2 = estimate_dtdb_l2(original, model)  # L2 distance to decision boundary
    dtdb_inf = estimate_dtdb_linf(original, model)  # L-inf distance to decision boundary
    
    # Return features in the exact order expected by PQP model:
    # [SSIM, PSNR, UQI, VIF, LPIPS, AD_2, AD_inf, DTDB_2, DTDB_inf]
    return np.array([ssim, psnr, uqi, vif, lpips_score, ad_2, ad_inf, dtdb_2, dtdb_inf])
```

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2025pglp,
  title={Perception-Guided Large Perturbation Attacks against Adversarial Training of Neural Networks},
  author={Li, Xiaowen and Zhao, Wenwei and Duan, Rui and Liu, Yao and Lu, Zhuo},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  note={Under review}
}
``` -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.


<!--## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: li33@usf.edu

## Acknowledgments

- This work was supported by [funding sources]
- We thank the reviewers for their valuable feedback
- PQP models were trained using human study data collected under IRB approval

## FAQ

**Q: Can you provide the raw human study data?**  
A: No, due to IRB regulations, we cannot release individual participant responses. The pre-trained PQP models are released instead.

**Q: Do I need to train my own PQP model?**  
A: No, you can use our pre-trained models for MNIST, CIFAR-10, and ImageNet. Training your own requires conducting a human study with IRB approval.

**Q: Can PGLP attacks be used against any model?**  
A: PGLP attacks are designed for black-box scenarios where you have access to the training dataset but not the target model. They work best against adversarially trained models.

**Q: How long does a PGLP attack take?**  
A: PGLP-S: ~10-100 iterations (seconds to minutes). PGLP-C: ~50-150 iterations total (minutes). Runtime depends on image size and model complexity.

**Q: Can I use PGLP for non-research purposes?**  
A: This code is provided for research and educational purposes only. Please use responsibly and ethically.-->
