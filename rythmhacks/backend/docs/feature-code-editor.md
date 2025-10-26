# Feature Documentation: Code Editor Integration

## Overview
Monaco Editor (VS Code engine) integration for writing, editing, and executing Python code for custom ML pipelines, data preprocessing, and model customization. Includes syntax highlighting, IntelliSense, and collaborative editing capabilities.

---

## Frontend Implementation

### Monaco Editor Setup

#### Installation

```bash
npm install @monaco-editor/react monaco-editor
```

#### Component Structure

```javascript
// frontend/src/components/CodeEditor.jsx
import React, { useState, useRef, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import './CodeEditor.css'

export default function CodeEditor({ 
  initialCode = '', 
  language = 'python',
  onExecute,
  readOnly = false,
  sessionId 
}) {
  const [code, setCode] = useState(initialCode)
  const [output, setOutput] = useState('')
  const [executing, setExecuting] = useState(false)
  const [theme, setTheme] = useState('vs-dark')
  const editorRef = useRef(null)
  const [ws, setWs] = useState(null)
  const [collaborators, setCollaborators] = useState([])

  // Monaco editor configuration
  const editorOptions = {
    minimap: { enabled: true },
    fontSize: 14,
    lineNumbers: 'on',
    roundedSelection: false,
    scrollBeyondLastLine: false,
    readOnly: readOnly,
    automaticLayout: true,
    tabSize: 4,
    wordWrap: 'on',
    formatOnPaste: true,
    formatOnType: true,
    suggestOnTriggerCharacters: true,
    quickSuggestions: true,
    snippetSuggestions: 'inline',
    folding: true,
    foldingStrategy: 'indentation',
    showFoldingControls: 'always',
    scrollbar: {
      vertical: 'visible',
      horizontal: 'visible',
      useShadows: false,
    }
  }

  // Initialize editor
  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor

    // Configure Python language features
    monaco.languages.registerCompletionItemProvider('python', {
      provideCompletionItems: (model, position) => {
        return {
          suggestions: getPythonSuggestions(monaco, model, position)
        }
      }
    })

    // Add custom keybindings
    editor.addCommand(
      monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter,
      () => handleExecute()
    )

    // Setup collaborative editing if session provided
    if (sessionId) {
      setupCollaborativeEditing(editor, monaco)
    }
  }

  // Get Python autocomplete suggestions
  const getPythonSuggestions = (monaco, model, position) => {
    const word = model.getWordUntilPosition(position)
    const range = {
      startLineNumber: position.lineNumber,
      endLineNumber: position.lineNumber,
      startColumn: word.startColumn,
      endColumn: word.endColumn
    }

    // ML library suggestions
    const mlLibraries = [
      { label: 'import numpy as np', insertText: 'import numpy as np' },
      { label: 'import pandas as pd', insertText: 'import pandas as pd' },
      { label: 'import torch', insertText: 'import torch' },
      { label: 'import torch.nn as nn', insertText: 'import torch.nn as nn' },
      { label: 'from sklearn.model_selection import train_test_split', insertText: 'from sklearn.model_selection import train_test_split' },
    ]

    // PyTorch snippets
    const torchSnippets = [
      {
        label: 'nn.Module class',
        insertText: 'class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        $0\n    \n    def forward(self, x):\n        return x',
        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
      },
      {
        label: 'training loop',
        insertText: 'for epoch in range(num_epochs):\n    for batch in dataloader:\n        optimizer.zero_grad()\n        output = model(batch)\n        loss = criterion(output, target)\n        loss.backward()\n        optimizer.step()',
        insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
      }
    ]

    return [
      ...mlLibraries.map(item => ({
        ...item,
        kind: monaco.languages.CompletionItemKind.Module,
        range: range
      })),
      ...torchSnippets.map(item => ({
        ...item,
        kind: monaco.languages.CompletionItemKind.Snippet,
        range: range
      }))
    ]
  }

  // Execute code
  const handleExecute = async () => {
    setExecuting(true)
    setOutput('Executing...\n')

    try {
      const response = await fetch('http://localhost:8000/api/code/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: code,
          language: language,
          session_id: sessionId
        })
      })

      const result = await response.json()

      if (result.success) {
        setOutput(prev => prev + '\n' + result.output)
        if (onExecute) {
          onExecute(result)
        }
      } else {
        setOutput(prev => prev + '\nError:\n' + result.error)
      }
    } catch (error) {
      setOutput(prev => prev + '\nFailed to execute: ' + error.message)
    } finally {
      setExecuting(false)
    }
  }

  // Setup collaborative editing via WebSocket
  const setupCollaborativeEditing = (editor, monaco) => {
    const websocket = new WebSocket(
      `ws://localhost:8000/ws/code/${sessionId}`
    )

    websocket.onopen = () => {
      console.log('Connected to collaborative editing')
    }

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'code_change') {
        // Apply remote changes
        if (data.user_id !== getCurrentUserId()) {
          const currentValue = editor.getValue()
          if (currentValue !== data.code) {
            editor.setValue(data.code)
          }
        }
      } else if (data.type === 'cursor_position') {
        // Show collaborator cursors
        updateCollaboratorCursor(data)
      } else if (data.type === 'user_joined') {
        setCollaborators(prev => [...prev, data.user])
      } else if (data.type === 'user_left') {
        setCollaborators(prev => prev.filter(u => u.id !== data.user.id))
      }
    }

    // Send code changes
    editor.onDidChangeModelContent((e) => {
      const newCode = editor.getValue()
      setCode(newCode)

      if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
          type: 'code_change',
          code: newCode,
          changes: e.changes
        }))
      }
    })

    // Send cursor position
    editor.onDidChangeCursorPosition((e) => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
          type: 'cursor_position',
          position: e.position
        }))
      }
    })

    setWs(websocket)

    return () => {
      if (websocket) {
        websocket.close()
      }
    }
  }

  const updateCollaboratorCursor = (data) => {
    // Visual indicator for other users' cursors
    // Implementation would add decorations to editor
  }

  // Save code to backend
  const handleSave = async () => {
    try {
      await fetch(`http://localhost:8000/api/code/save/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code })
      })
      alert('Code saved successfully')
    } catch (error) {
      alert('Failed to save code')
    }
  }

  return (
    <div className="code-editor-container">
      <div className="editor-toolbar">
        <div className="toolbar-left">
          <h3>Code Editor</h3>
          <select 
            value={language} 
            onChange={(e) => setLanguage(e.target.value)}
            className="language-select"
          >
            <option value="python">Python</option>
            <option value="javascript">JavaScript</option>
            <option value="json">JSON</option>
          </select>
          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            className="theme-select"
          >
            <option value="vs-dark">Dark</option>
            <option value="vs-light">Light</option>
            <option value="hc-black">High Contrast</option>
          </select>
        </div>

        <div className="toolbar-right">
          {collaborators.length > 0 && (
            <div className="collaborators">
              {collaborators.map(user => (
                <span key={user.id} className="collaborator-badge" title={user.name}>
                  {user.name.charAt(0)}
                </span>
              ))}
            </div>
          )}
          <button onClick={handleSave} className="btn-save">
            💾 Save
          </button>
          <button 
            onClick={handleExecute} 
            className="btn-execute"
            disabled={executing}
          >
            {executing ? '⏳ Running...' : '▶️ Run (Ctrl+Enter)'}
          </button>
        </div>
      </div>

      <div className="editor-main">
        <div className="editor-pane">
          <Editor
            height="100%"
            language={language}
            value={code}
            onChange={(value) => setCode(value || '')}
            onMount={handleEditorDidMount}
            theme={theme}
            options={editorOptions}
          />
        </div>

        <div className="output-pane">
          <div className="output-header">
            <h4>Output</h4>
            <button onClick={() => setOutput('')} className="btn-clear">
              Clear
            </button>
          </div>
          <pre className="output-content">{output || 'No output yet. Run code to see results.'}</pre>
        </div>
      </div>
    </div>
  )
}

// Helper function - would come from auth context
const getCurrentUserId = () => {
  return 'user_' + Math.random().toString(36).substr(2, 9)
}
```

---

## Backend Integration

### Code Execution Engine

```python
# backend/app/api/routes/code.py
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
import io
import contextlib
import traceback
import ast
import timeout_decorator
from pathlib import Path

router = APIRouter()

class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    session_id: Optional[str] = None
    timeout: int = 30  # seconds
    capture_output: bool = True

class CodeExecutionResponse(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float
    variables: Optional[dict] = None

@router.post("/api/code/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """
    Execute Python code in a sandboxed environment
    
    Security considerations:
    - Timeout to prevent infinite loops
    - Restricted imports (no os, subprocess, etc.)
    - Limited file system access
    - Resource limits
    """
    import time
    start_time = time.time()
    
    if request.language != "python":
        raise HTTPException(400, "Only Python is currently supported")
    
    # Validate code safety
    try:
        validate_code_safety(request.code)
    except ValueError as e:
        return CodeExecutionResponse(
            success=False,
            output="",
            error=f"Code validation failed: {str(e)}",
            execution_time=0
        )
    
    # Execute code
    try:
        output, variables = execute_python_code(
            request.code, 
            timeout=request.timeout
        )
        
        execution_time = time.time() - start_time
        
        return CodeExecutionResponse(
            success=True,
            output=output,
            execution_time=execution_time,
            variables=variables
        )
    
    except TimeoutError:
        return CodeExecutionResponse(
            success=False,
            output="",
            error="Execution timed out",
            execution_time=request.timeout
        )
    
    except Exception as e:
        return CodeExecutionResponse(
            success=False,
            output="",
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            execution_time=time.time() - start_time
        )

def validate_code_safety(code: str):
    """
    Validate code for dangerous operations
    
    Raises ValueError if code contains forbidden patterns
    """
    # Parse AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")
    
    # Forbidden imports
    forbidden_modules = {
        'os', 'subprocess', 'sys', 'eval', 'exec',
        'compile', '__import__', 'open', 'file'
    }
    
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    raise ValueError(f"Import of '{alias.name}' is not allowed")
        
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden_modules:
                raise ValueError(f"Import from '{node.module}' is not allowed")
        
        # Check function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {'eval', 'exec', 'compile', '__import__', 'open'}:
                    raise ValueError(f"Use of '{node.func.id}' is not allowed")

@timeout_decorator.timeout(30)
def execute_python_code(code: str, timeout: int = 30):
    """
    Execute Python code and capture output
    
    Returns (output, variables)
    """
    # Redirect stdout/stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    # Create restricted globals
    safe_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
        },
        # Allow ML libraries
        'np': __import__('numpy'),
        'pd': __import__('pandas'),
        'torch': __import__('torch'),
        'nn': __import__('torch.nn'),
        'plt': __import__('matplotlib.pyplot'),
    }
    
    local_vars = {}
    
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            exec(code, safe_globals, local_vars)
        except Exception as e:
            raise
    
    # Capture output
    output = stdout.getvalue()
    if stderr.getvalue():
        output += "\nStderr:\n" + stderr.getvalue()
    
    # Extract variables (exclude builtins and modules)
    variables = {
        k: str(v) for k, v in local_vars.items()
        if not k.startswith('_') and not callable(v)
    }
    
    return output, variables

@router.post("/api/code/save/{session_id}")
async def save_code(session_id: str, code: dict):
    """Save code to session"""
    from backend.app.models.code_session import CodeSession
    
    session = db.query(CodeSession).filter_by(id=session_id).first()
    if not session:
        session = CodeSession(id=session_id, code=code['code'])
        db.add(session)
    else:
        session.code = code['code']
        session.updated_at = datetime.utcnow()
    
    db.commit()
    return {'status': 'saved'}

@router.get("/api/code/load/{session_id}")
async def load_code(session_id: str):
    """Load code from session"""
    from backend.app.models.code_session import CodeSession
    
    session = db.query(CodeSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")
    
    return {'code': session.code}

# WebSocket for collaborative editing
class CodeCollaborationManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        
        # Notify others
        await self.broadcast(session_id, {
            'type': 'user_joined',
            'user': {'id': user_id, 'name': get_user_name(user_id)}
        }, exclude=websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
    
    async def broadcast(self, session_id: str, message: dict, exclude: WebSocket = None):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                if connection != exclude:
                    try:
                        await connection.send_json(message)
                    except:
                        pass

code_manager = CodeCollaborationManager()

@router.websocket("/ws/code/{session_id}")
async def code_websocket(websocket: WebSocket, session_id: str, user_id: str = None):
    """WebSocket for collaborative code editing"""
    user_id = user_id or generate_temp_user_id()
    
    await code_manager.connect(websocket, session_id, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Add user_id to message
            message['user_id'] = user_id
            
            # Broadcast to others
            await code_manager.broadcast(session_id, message, exclude=websocket)
    
    except WebSocketDisconnect:
        code_manager.disconnect(websocket, session_id)
        await code_manager.broadcast(session_id, {
            'type': 'user_left',
            'user': {'id': user_id}
        })
```

---

## Database Models

```python
# backend/app/models/code_session.py
from sqlalchemy import Column, String, Text, DateTime
from datetime import datetime

class CodeSession(Base):
    __tablename__ = "code_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    code = Column(Text, nullable=False, default="")
    language = Column(String, default="python")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

---

## Styling

```css
/* frontend/src/components/CodeEditor.css */
.code-editor-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #1e1e1e;
  border-radius: 8px;
  overflow: hidden;
}

.editor-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #2d2d2d;
  border-bottom: 1px solid #3e3e3e;
}

.toolbar-left,
.toolbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.toolbar-left h3 {
  margin: 0;
  color: #fff;
  font-size: 14px;
}

.language-select,
.theme-select {
  padding: 6px 10px;
  background: #3c3c3c;
  color: #fff;
  border: 1px solid #555;
  border-radius: 4px;
  font-size: 12px;
}

.collaborators {
  display: flex;
  gap: 4px;
}

.collaborator-badge {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #4682B4;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
}

.btn-save,
.btn-execute {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-save {
  background: #4682B4;
  color: white;
}

.btn-execute {
  background: #28a745;
  color: white;
}

.btn-execute:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.editor-main {
  display: grid;
  grid-template-columns: 1fr 400px;
  flex: 1;
  overflow: hidden;
}

.editor-pane {
  border-right: 1px solid #3e3e3e;
}

.output-pane {
  background: #1e1e1e;
  display: flex;
  flex-direction: column;
}

.output-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: #2d2d2d;
  border-bottom: 1px solid #3e3e3e;
}

.output-header h4 {
  margin: 0;
  color: #fff;
  font-size: 13px;
}

.btn-clear {
  padding: 4px 12px;
  background: transparent;
  color: #fff;
  border: 1px solid #555;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
}

.output-content {
  flex: 1;
  padding: 16px;
  margin: 0;
  color: #d4d4d4;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  overflow-y: auto;
  white-space: pre-wrap;
}
```

---

## Future Enhancements

1. **Jupyter Notebook Integration**: Execute cells independently
2. **Git Integration**: Version control for code
3. **Code Templates**: Pre-built ML pipeline templates
4. **Debugging**: Breakpoints and step-through debugging
5. **Code Formatting**: Auto-format with Black/autopep8
6. **Linting**: Real-time PEP8 checking
7. **Package Management**: Install packages from editor
8. **Export**: Download code as .py file
9. **Code Sharing**: Share snippets via URL
10. **AI Assistance**: Code suggestions and bug fixes
## MVP Additions (Missing)

- Reusable component API: props for `language`, `value`, `onChange`, `height`.
- Placement: top of model visualization pages (Decision Tree, Regression, NN).
- Basic toolbar: copy, clear, format (optional), download snippet.
- Persistence: autosave to localStorage per route/dataset id.
- Accessibility: keyboard shortcuts (Ctrl/Cmd+S to save, Tab handling).
## Collaborative Editing (MVP, two users)

- Room model: one room per route + datasetId (e.g., `decision-tree:ds_123`).
- Presence: show remote cursor and selection with user color and name.
- Live changes: broadcast `editor_change` events with coarse ops (range + text). Throttle to 20–30 updates/sec.
- Conflict handling: last-writer-wins with version counter; periodic full-buffer reconciliation via `save` events.
- Autosave: every 10s and on blur; backend persists latest content by room.
- Offline: buffer stored locally; upon reconnect, send `save {version, content}` and accept server ack or overwrite.

### WebSocket Protocol (sketch)

- `join {roomId, user}` → ack {roomId, users, version, content}
- `presence {cursor:{line,ch}, selection:{anchor,head}}`
- `editor_change {version, range:{from,to}, text}` → ack {version}
- `save {version, content}` → ack {version}
- `leave {roomId}`

### Frontend Integration

- Library: Monaco or CodeMirror 6; start with CodeMirror for lighter bundle.
- Adapter maps editor transactions → WS `editor_change` messages and applies remote changes via editor API.
- State stored in React context keyed by room; reconnect logic retries with backoff.
- Feature flags: `REACT_APP_COLLAB=on` to enable; falls back to local-only if WS unavailable.

### Backend Integration

- Tech: FastAPI + `websockets` or Starlette WS endpoints.
- In-memory room state for MVP: `{version, content, users}`; persist latest `content` to disk under `models/rooms/{roomId}.txt`.
- Broadcast changes to all users in room; validate monotonically increasing version.

## Toolbar and Actions

- Buttons: Copy, Clear, Download, Save. Optional: Format (prettier/black if language matches).
- Keyboard: Ctrl/Cmd+S triggers `save`.
