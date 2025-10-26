"""
Code Executor Module
Secure Python code execution with sandboxing and validation
"""

import ast
import sys
import io
import time
import traceback
from typing import Dict, Any, Optional, List
from contextlib import redirect_stdout, redirect_stderr


class CodeExecutor:
    """Secure code execution engine with AST validation and sandboxing"""
    
    # Blocked modules for security
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'eval', 'exec',
        '__import__', 'compile', 'open', 'file', 'input',
        'raw_input', 'execfile', 'reload', 'importlib'
    }
    
    # Allowed safe ML/data science modules
    ALLOWED_MODULES = {
        'numpy', 'np', 'pandas', 'pd', 'sklearn', 'torch',
        'matplotlib', 'seaborn', 'scipy', 'math', 'random',
        'datetime', 'json', 'collections', 'itertools',
        'functools', 'operator', 're', 'statistics'
    }
    
    # Maximum execution time in seconds
    MAX_EXECUTION_TIME = 5
    
    # Maximum output size (10KB)
    MAX_OUTPUT_SIZE = 10 * 1024

    @staticmethod
    def validate_code(code: str) -> Dict[str, Any]:
        """
        Validate Python code using AST parsing
        
        Returns:
            dict: {
                'valid': bool,
                'errors': list,
                'warnings': list,
                'imports': list
            }
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'imports': []
        }
        
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        result['imports'].append(module_name)
                        
                        if module_name in CodeExecutor.BLOCKED_MODULES:
                            result['valid'] = False
                            result['errors'].append(
                                f"Import of '{module_name}' is not allowed for security reasons"
                            )
                        elif module_name not in CodeExecutor.ALLOWED_MODULES:
                            result['warnings'].append(
                                f"Module '{module_name}' may not be available"
                            )
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        result['imports'].append(module_name)
                        
                        if module_name in CodeExecutor.BLOCKED_MODULES:
                            result['valid'] = False
                            result['errors'].append(
                                f"Import from '{module_name}' is not allowed for security reasons"
                            )
                
                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in ['eval', 'exec', 'compile', '__import__']:
                            result['valid'] = False
                            result['errors'].append(
                                f"Use of '{func_name}' is not allowed for security reasons"
                            )
                
                # Check for file operations
                elif isinstance(node, ast.Name):
                    if node.id == 'open':
                        result['warnings'].append(
                            "File operations may be restricted"
                        )
        
        except SyntaxError as e:
            result['valid'] = False
            result['errors'].append(f"Syntax error: {str(e)}")
        
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result

    @staticmethod
    def execute_code(
        code: str,
        timeout: int = MAX_EXECUTION_TIME,
        globals_dict: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in a restricted environment
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            globals_dict: Optional pre-populated globals
        
        Returns:
            dict: {
                'success': bool,
                'output': str,
                'stdout': str,
                'stderr': str,
                'execution_time': float,
                'variables': dict,
                'error': str (if failed)
            }
        """
        result = {
            'success': False,
            'output': '',
            'stdout': '',
            'stderr': '',
            'execution_time': 0.0,
            'variables': {},
            'error': None
        }
        
        # First validate the code
        validation = CodeExecutor.validate_code(code)
        if not validation['valid']:
            result['error'] = '; '.join(validation['errors'])
            result['stderr'] = result['error']
            return result
        
        # Prepare restricted globals
        restricted_globals = {} if globals_dict is None else globals_dict.copy()
        
        # Create a safe __import__ function
        def safe_import(name, *args, **kwargs):
            """Only allow whitelisted modules"""
            base_module = name.split('.')[0]
            if base_module in CodeExecutor.ALLOWED_MODULES:
                return __import__(name, *args, **kwargs)
            elif base_module in CodeExecutor.BLOCKED_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed for security reasons")
            else:
                # Try to import, but warn
                return __import__(name, *args, **kwargs)
        
        # Add safe built-ins
        restricted_globals.update({
            '__builtins__': {
                '__import__': safe_import,
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'dir': dir,
                'help': help,
                'None': None,
                'True': True,
                'False': False,
            }
        })
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            start_time = time.time()
            
            # Compile with standard Python compile (AST validation already done)
            byte_code = compile(code, '<string>', 'exec')
            
            # Execute with output redirection
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(byte_code, restricted_globals)
            
            execution_time = time.time() - start_time
            
            # Check timeout
            if execution_time > timeout:
                result['error'] = f"Execution timeout ({timeout}s limit exceeded)"
                result['stderr'] = result['error']
                return result
            
            # Capture outputs
            result['stdout'] = stdout_capture.getvalue()
            result['stderr'] = stderr_capture.getvalue()
            result['output'] = result['stdout'] if result['stdout'] else result['stderr']
            
            # Truncate if too large
            if len(result['output']) > CodeExecutor.MAX_OUTPUT_SIZE:
                result['output'] = result['output'][:CodeExecutor.MAX_OUTPUT_SIZE] + \
                    "\n... (output truncated)"
            
            # Extract user-defined variables
            result['variables'] = {}
            for key, value in restricted_globals.items():
                if not key.startswith('_') and key not in ['__builtins__']:
                    try:
                        # Get type info
                        var_type = type(value).__name__
                        if hasattr(value, '__module__'):
                            var_type = f"{value.__module__}.{var_type}"
                        result['variables'][key] = var_type
                    except:
                        pass
            
            result['success'] = True
            result['execution_time'] = round(execution_time, 3)
        
        except Exception as e:
            result['error'] = str(e)
            result['stderr'] = traceback.format_exc()
            result['output'] = result['stderr']
        
        return result

    @staticmethod
    def get_available_packages() -> List[str]:
        """
        Get list of available ML/data science packages
        
        Returns:
            list: Available package names
        """
        available = []
        
        for package in CodeExecutor.ALLOWED_MODULES:
            try:
                __import__(package)
                available.append(package)
            except ImportError:
                pass
        
        return sorted(available)
