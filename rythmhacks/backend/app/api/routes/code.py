"""
Code Execution Routes
API endpoints for executing and validating Python code
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.core.code_executor import CodeExecutor

router = APIRouter(prefix="/api/code")


class CodeExecutionRequest(BaseModel):
    """Request model for code execution"""
    code: str = Field(..., description="Python code to execute")
    language: str = Field(default="python", description="Programming language (only python supported)")
    timeout: Optional[int] = Field(default=5, ge=1, le=10, description="Execution timeout in seconds")
    session_id: Optional[str] = Field(default=None, description="Session ID for context sharing")


class CodeValidationRequest(BaseModel):
    """Request model for code validation"""
    code: str = Field(..., description="Python code to validate")


class CodeExecutionResponse(BaseModel):
    """Response model for code execution"""
    success: bool
    output: str
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    variables: Dict[str, str] = {}
    error: Optional[str] = None


class CodeValidationResponse(BaseModel):
    """Response model for code validation"""
    valid: bool
    errors: list = []
    warnings: list = []
    imports: list = []


@router.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """
    Execute Python code in a secure sandbox environment
    
    Security features:
    - AST validation to prevent dangerous operations
    - Restricted module imports (ML libraries allowed)
    - No file system access
    - Execution timeout (default 5s, max 10s)
    - Output size limit (10KB)
    
    Returns:
        CodeExecutionResponse with execution results
    
    Example:
        ```python
        POST /api/code/execute
        {
            "code": "import numpy as np\\nprint(np.array([1,2,3]).mean())",
            "timeout": 5
        }
        ```
    """
    if request.language != "python":
        raise HTTPException(
            status_code=400,
            detail="Only Python language is currently supported"
        )
    
    if not request.code.strip():
        raise HTTPException(
            status_code=400,
            detail="Code cannot be empty"
        )
    
    try:
        # Execute code with sandbox
        result = CodeExecutor.execute_code(
            code=request.code,
            timeout=request.timeout
        )
        
        return CodeExecutionResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Code execution failed: {str(e)}"
        )


@router.post("/validate", response_model=CodeValidationResponse)
async def validate_code(request: CodeValidationRequest):
    """
    Validate Python code without executing it
    
    Performs:
    - Syntax checking
    - AST parsing
    - Import validation
    - Security checks
    
    Returns:
        CodeValidationResponse with validation results
    
    Example:
        ```python
        POST /api/code/validate
        {
            "code": "import numpy as np\\nx = [1, 2, 3]"
        }
        ```
    """
    if not request.code.strip():
        raise HTTPException(
            status_code=400,
            detail="Code cannot be empty"
        )
    
    try:
        # Validate code
        result = CodeExecutor.validate_code(request.code)
        
        return CodeValidationResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Code validation failed: {str(e)}"
        )


@router.get("/packages")
async def get_available_packages():
    """
    Get list of available ML/data science packages
    
    Returns:
        List of installed and available packages
    
    Example:
        ```python
        GET /api/code/packages
        ```
    """
    try:
        packages = CodeExecutor.get_available_packages()
        
        return {
            "packages": packages,
            "total": len(packages),
            "categories": {
                "numerical": ["numpy"],
                "data_manipulation": ["pandas"],
                "machine_learning": ["sklearn", "torch"],
                "visualization": ["matplotlib", "seaborn"],
                "scientific": ["scipy"]
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve packages: {str(e)}"
        )


@router.get("/health")
async def code_execution_health():
    """
    Health check for code execution service
    
    Returns:
        Service status and configuration
    """
    return {
        "status": "operational",
        "service": "Code Execution Engine",
        "features": {
            "languages": ["python"],
            "max_timeout": CodeExecutor.MAX_EXECUTION_TIME,
            "max_output_size": CodeExecutor.MAX_OUTPUT_SIZE,
            "security": "RestrictedPython + AST validation"
        },
        "restrictions": {
            "blocked_modules": list(CodeExecutor.BLOCKED_MODULES),
            "allowed_modules": list(CodeExecutor.ALLOWED_MODULES)
        }
    }
