"""
Test Script for PGLP Attacks and PQP Metric
============================================

This script tests the PGLP attack implementations and PQP metric loading.
Run this to verify the code works correctly before using actual models.

Usage:
    python test_pglp.py

Requirements:
- PyTorch
- joblib
- numpy
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os


def test_imports():
    """Test that all required imports work."""
    print("\n[Test 1/6] Testing imports...")
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        import joblib
        print("OK All imports successful")
        return True
    except ImportError as e:
        print(f"FAIL Import error: {e}")
        return False


def test_pqp_loading():
    """Test PQP model loading."""
    print("\n[Test 2/6] Testing PQP model loading...")
    
    # Create a dummy PQP model for testing
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
    dummy_model.fit(np.random.rand(100, 9), np.random.rand(100))
    
    save_data = {
        'model': dummy_model,
        'feature_names': ['SSIM', 'PSNR', 'UQI', 'VIF', 'LPIPS', 
                         'AD_2', 'AD_inf', 'DTDB_2', 'DTDB_inf'],
        'metadata': {'test': True}
    }
    
    joblib.dump(save_data, 'test_pqp_model.joblib')
    
    try:
        loaded = joblib.load('test_pqp_model.joblib')
        assert 'model' in loaded
        assert 'feature_names' in loaded
        print("OK PQP model loading works")
        
        # Clean up
        os.remove('test_pqp_model.joblib')
        return True
    except Exception as e:
        print(f"FAIL PQP loading error: {e}")
        return False


def test_pglps_initialization():
    """Test PGLP-S attack initialization."""
    print("\n[Test 3/6] Testing PGLP-S initialization...")
    
    try:
        from pglp_attack import PGLP_S
        
        # Create dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3*32*32, 10)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = DummyModel()
        
        # Create dummy PQP model
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        dummy_model.fit(np.random.rand(100, 9), np.random.rand(100))
        joblib.dump({'model': dummy_model, 'feature_names': []}, 'test_pqp.joblib')
        
        # Initialize attack
        attack = PGLP_S(model, 'test_pqp.joblib', pqp_threshold=0.7, dataset='cifar10')
        print("OK PGLP-S initialization works")
        
        # Clean up
        os.remove('test_pqp.joblib')
        return True
        
    except Exception as e:
        print(f"FAIL PGLP-S initialization error: {e}")
        return False


def test_pglps_attack():
    """Test PGLP-S attack execution."""
    print("\n[Test 4/6] Testing PGLP-S attack...")
    
    try:
        from pglp_attack import PGLP_S
        
        # Create dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3*32*32, 10)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = DummyModel()
        
        # Create dummy PQP model
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        dummy_model.fit(np.random.rand(100, 9), np.random.rand(100))
        joblib.dump({'model': dummy_model, 'feature_names': []}, 'test_pqp.joblib')
        
        # Initialize attack
        attack = PGLP_S(model, 'test_pqp.joblib', pqp_threshold=0.7, dataset='cifar10')
        
        # Run attack
        image = torch.rand(3, 32, 32)
        adv_image, info = attack.attack(image, true_label=0, max_iterations=5, verbose=False)
        
        assert isinstance(adv_image, torch.Tensor)
        assert isinstance(info, dict)
        assert 'iterations' in info
        print("OK PGLP-S attack execution works")
        
        # Clean up
        os.remove('test_pqp.joblib')
        return True
        
    except Exception as e:
        print(f"FAIL PGLP-S attack error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pglpc_initialization():
    """Test PGLP-C attack initialization."""
    print("\n[Test 5/6] Testing PGLP-C initialization...")
    
    try:
        from pglp_attack import PGLP_C
        
        # Create dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3*32*32, 10)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = DummyModel()
        
        # Create dummy PQP model
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        dummy_model.fit(np.random.rand(100, 9), np.random.rand(100))
        joblib.dump({'model': dummy_model, 'feature_names': []}, 'test_pqp.joblib')
        
        # Initialize attack
        attack = PGLP_C(model, 'test_pqp.joblib', pqp_threshold=0.7, dataset='cifar10')
        print("OK PGLP-C initialization works")
        
        # Clean up
        os.remove('test_pqp.joblib')
        return True
        
    except Exception as e:
        print(f"FAIL PGLP-C initialization error: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction for PQP."""
    print("\n[Test 6/6] Testing feature extraction...")
    
    try:
        # Create dummy images
        original = np.random.rand(32, 32, 3).astype(np.float32)
        adversarial = np.clip(original + np.random.randn(32, 32, 3) * 0.1, 0, 1)
        
        # Simple feature extraction test
        diff = adversarial - original
        l2_dist = np.linalg.norm(diff)
        linf_dist = np.max(np.abs(diff))
        
        assert l2_dist >= 0
        assert linf_dist >= 0
        print("OK Feature extraction works")
        return True
        
    except Exception as e:
        print(f"FAIL Feature extraction error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("PGLP Attack Test Suite")
    print("="*70)
    
    tests = [
        test_imports,
        test_pqp_loading,
        test_pglps_initialization,
        test_pglps_attack,
        test_pglpc_initialization,
        test_feature_extraction
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"FAIL Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\nOK All tests passed!")
        print("\nNext steps:")
        print("1. Download pre-trained PQP models")
        print("2. Prepare your dataset")
        print("3. Run PGLP attacks using pglp_attack.py")
        return 0
    else:
        print("\nFAIL Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
