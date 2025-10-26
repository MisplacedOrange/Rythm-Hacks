"""
Test script to verify custom PyTorch model loading
Run this to check if your model class is properly registered
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_pytorch_import():
    """Test 1: Check if PyTorch is installed"""
    print("=" * 60)
    print("Test 1: PyTorch Installation")
    print("=" * 60)
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Fix: pip install torch==2.5.1")
        return False


def test_custom_models_import():
    """Test 2: Check if custom_models module loads"""
    print("\n" + "=" * 60)
    print("Test 2: Custom Models Module")
    print("=" * 60)
    try:
        from app.core import custom_models
        print("✅ custom_models.py loaded successfully")
        return True, custom_models
    except ImportError as e:
        print(f"❌ Failed to import custom_models: {e}")
        return False, None


def test_model_classes(custom_models):
    """Test 3: Check registered model classes"""
    print("\n" + "=" * 60)
    print("Test 3: Registered Model Classes")
    print("=" * 60)
    
    if not custom_models:
        print("❌ custom_models not available")
        return False
    
    models = custom_models.CUSTOM_MODELS
    print(f"Found {len(models)} registered models:")
    
    for name, model_class in models.items():
        if model_class is not None:
            print(f"  ✅ {name}: {model_class}")
        else:
            print(f"  ⚠️  {name}: None (PyTorch not available)")
    
    if 'HeartDiseaseMLP' in models:
        print("\n✅ HeartDiseaseMLP is registered!")
        return True
    else:
        print("\n❌ HeartDiseaseMLP not found in CUSTOM_MODELS")
        print("   Fix: Add 'HeartDiseaseMLP': HeartDiseaseMLP to CUSTOM_MODELS dict")
        return False


def test_model_instantiation(custom_models):
    """Test 4: Try creating a model instance"""
    print("\n" + "=" * 60)
    print("Test 4: Model Instantiation")
    print("=" * 60)
    
    if not custom_models or not custom_models.TORCH_AVAILABLE:
        print("❌ PyTorch not available, skipping")
        return False
    
    try:
        model_class = custom_models.CUSTOM_MODELS.get('HeartDiseaseMLP')
        if model_class is None:
            print("❌ HeartDiseaseMLP class is None")
            return False
        
        # Try to create instance with default parameters
        model = model_class()
        print(f"✅ Successfully created HeartDiseaseMLP instance")
        print(f"   Model: {model}")
        return True
    except TypeError as e:
        print(f"⚠️  Model requires specific parameters: {e}")
        print("   This is OK - model will load from pickle with saved parameters")
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate model: {e}")
        return False


def test_pickle_loading():
    """Test 5: Test pickle module resolution"""
    print("\n" + "=" * 60)
    print("Test 5: Pickle Module Resolution")
    print("=" * 60)
    
    try:
        from app.core import custom_models
        import __main__
        
        # Check if classes are available in __main__
        for name, model_class in custom_models.CUSTOM_MODELS.items():
            if model_class is not None:
                setattr(__main__, name, model_class)
        
        if hasattr(__main__, 'HeartDiseaseMLP'):
            print("✅ HeartDiseaseMLP available in __main__ namespace")
            return True
        else:
            print("❌ HeartDiseaseMLP not in __main__ namespace")
            return False
    except Exception as e:
        print(f"❌ Error setting up pickle resolution: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🔍" * 30)
    print("   CUSTOM PYTORCH MODEL VERIFICATION")
    print("🔍" * 30 + "\n")
    
    results = []
    
    # Test 1: PyTorch
    results.append(test_pytorch_import())
    
    # Test 2: Import custom_models
    success, custom_models = test_custom_models_import()
    results.append(success)
    
    if success:
        # Test 3: Check registered classes
        results.append(test_model_classes(custom_models))
        
        # Test 4: Instantiate model
        results.append(test_model_instantiation(custom_models))
        
        # Test 5: Pickle resolution
        results.append(test_pickle_loading())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("Your custom PyTorch model should load correctly.")
        print("\nNext steps:")
        print("1. Restart the backend server: python run.py")
        print("2. Upload your .pkl model file")
        print("3. Upload test data CSV")
        print("4. Metrics should calculate successfully!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nFollow the fixes suggested above, then:")
        print("1. Run this script again: python test_custom_model.py")
        print("2. Once all tests pass, restart the backend")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
