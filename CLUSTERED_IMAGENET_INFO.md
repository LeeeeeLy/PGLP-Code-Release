# Clustered ImageNet Dataset Information

## Overview

The PQP model (`pqp_imagenet.joblib`) is trained on **Clustered ImageNet**, a custom subset of standard ImageNet with only **10 coarse categories** as defined in the paper.

## Categories

| ID | Category | Description |
|----|----------|-------------|
| 0 | Spider | Arachnids |
| 1 | Beetle | Insects (beetles) |
| 2 | Bird | Birds |
| 3 | Butterfly | Insects (butterflies, moths) |
| 4 | Cat | Felines |
| 5 | Dog | Canines |
| 6 | Fish | Aquatic animals |
| 7 | Lizard | Reptiles (lizards) |
| 8 | Monkey | Primates |
| 9 | Snake | Reptiles (snakes) |

## How to Generate Clustered ImageNet

We provide a ready-to-use script for generating Clustered ImageNet:

**Repository**: https://github.com/LeeeeeLy/Clustered-ImageNet64-with-path-fixer

**Steps**:
1. Download ImageNet64 from [ImageNet Download Page](https://www.image-net.org/download.php)
2. Extract to `downloadeddata/train` and `downloadeddata/val`
3. Run `extractimages.py` to decode and save as PNG
4. Run `clustereddata.py` to cluster images into 10 categories



## Key Points

- **Total classes**: 10 (vs 1000 in full ImageNet)
- **Resolution**: 64×64 (for efficiency)
- **PQP features used**: [1,2,3,4,6,7,8] = SSIM, PSNR, UQI, VIF, AD_2, AD_inf, DTDB_2
- **MSE**: 0.035984
- **Correlation with human ratings**: 0.912232

## Note for Reviewers

The Clustered ImageNet differs from standard ImageNet:
- Fewer classes (10 vs 1000)
- Different data distribution
- Not directly comparable to standard ImageNet benchmarks

For fair comparison, all experiments in the paper use the same Clustered ImageNet subset.
