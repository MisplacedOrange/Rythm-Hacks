"""Core ML engines and utilities"""

# Import custom models at module level to make them globally available
# This ensures PyTorch can find custom classes during unpickling
try:
    from app.core.custom_models import *  # noqa: F401, F403
    from app.core.custom_models import CUSTOM_MODELS, TORCH_AVAILABLE
    
    if TORCH_AVAILABLE:
        print("✅ Custom PyTorch models loaded globally")
        print(f"   Available models: {list(CUSTOM_MODELS.keys())}")
    else:
        print("⚠️  PyTorch not available - custom models disabled")
        
except ImportError as e:
    print(f"❌ Failed to import custom models: {e}")

