"""
Quick test script for code execution functionality
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.code_executor import CodeExecutor


def test_basic_execution():
    """Test 1: Basic code execution"""
    print("\n=== Test 1: Basic Execution ===")
    code = """
print("Hello from Monaco Editor!")
x = 10 + 20
print(f"Result: {x}")
"""
    result = CodeExecutor.execute_code(code)
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['output']}")
    assert result['success'], "Basic execution should succeed"
    assert "Hello from Monaco Editor!" in result['output']
    print("✅ PASSED")


def test_numpy_pandas():
    """Test 2: NumPy/Pandas imports"""
    print("\n=== Test 2: NumPy/Pandas ===")
    code = """
import numpy as np
import pandas as pd

arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {arr.mean()}")

df = pd.DataFrame({'A': [1, 2, 3]})
print(df)
"""
    result = CodeExecutor.execute_code(code)
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['output']}")
    assert result['success'], "NumPy/Pandas execution should succeed"
    assert "Mean:" in result['output']
    print("✅ PASSED")


def test_security_validation():
    """Test 3: Security - blocked modules"""
    print("\n=== Test 3: Security Validation ===")
    code = """
import os
os.system('dir')
"""
    result = CodeExecutor.execute_code(code)
    print(f"Success: {result['success']}")
    print(f"Error: {result.get('error', 'None')}")
    assert not result['success'], "Security validation should block dangerous imports"
    assert 'os' in result.get('error', '').lower()
    print("✅ PASSED - Dangerous code blocked")


def test_validation_only():
    """Test 4: Code validation without execution"""
    print("\n=== Test 4: Validation Only ===")
    code = """
import numpy as np
x = np.array([1, 2, 3])
print(x.mean())
"""
    result = CodeExecutor.validate_code(code)
    print(f"Valid: {result['valid']}")
    print(f"Imports: {result['imports']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    assert result['valid'], "Valid code should pass validation"
    assert 'numpy' in result['imports']
    print("PASSED")


def test_sklearn():
    """Test 5: sklearn model training"""
    print("\n=== Test 5: Sklearn Model ===")
    code = """
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict([[6], [7]])
print(f"Predictions: {predictions}")
print(f"Coefficient: {model.coef_[0]:.2f}")
"""
    result = CodeExecutor.execute_code(code)
    print(f"Success: {result['success']}")
    print(f"Output:\n{result['output']}")
    print(f"Variables: {result.get('variables', {})}")
    assert result['success'], "Sklearn execution should succeed"
    assert "Predictions:" in result['output']
    print("PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("CODE EXECUTOR TEST SUITE")
    print("=" * 60)
    
    try:
        test_basic_execution()
        test_numpy_pandas()
        test_security_validation()
        test_validation_only()
        test_sklearn()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
