import React, { useState, useRef } from 'react'
import Editor from '@monaco-editor/react'
import './CodeEditor.css'

const DEFAULT_CODE = `# Python Code Editor - MediLytica
# Press Ctrl+Enter to execute

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Example: Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict([[6], [7]])
print(f"Predictions: {predictions}")
print(f"Coefficient: {model.coef_[0]:.2f}")
`

export default function CodeEditor({ modelId, sessionId }) {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [output, setOutput] = useState('')
  const [executing, setExecuting] = useState(false)
  const [theme, setTheme] = useState('vs-dark')
  const editorRef = useRef(null)

  // Monaco editor configuration
  const editorOptions = {
    minimap: { enabled: true },
    fontSize: 14,
    lineNumbers: 'on',
    roundedSelection: false,
    scrollBeyondLastLine: false,
    readOnly: false,
    automaticLayout: true,
    tabSize: 4,
    wordWrap: 'off',
    formatOnPaste: false,
    formatOnType: false,
    suggestOnTriggerCharacters: false,
    quickSuggestions: false,
    snippetSuggestions: 'none',
    folding: true,
    foldingStrategy: 'indentation',
    showFoldingControls: 'always',
    padding: { top: 10, bottom: 10 },
    lineNumbersMinChars: 4,
    glyphMargin: true,
    acceptSuggestionOnEnter: 'off',
    scrollbar: {
      vertical: 'visible',
      horizontal: 'visible',
      useShadows: false,
      verticalScrollbarSize: 14,
      horizontalScrollbarSize: 14,
    }
  }

  // Initialize editor and add keybindings
  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor

    // Add Ctrl+Enter keybinding to execute code
    editor.addCommand(
      monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter,
      () => handleExecute()
    )

    // Configure Python language features
    monaco.languages.registerCompletionItemProvider('python', {
      provideCompletionItems: (model, position) => {
        const word = model.getWordUntilPosition(position)
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn
        }

        // ML library suggestions
        const suggestions = [
          {
            label: 'import numpy as np',
            kind: monaco.languages.CompletionItemKind.Module,
            insertText: 'import numpy as np',
            range: range
          },
          {
            label: 'import pandas as pd',
            kind: monaco.languages.CompletionItemKind.Module,
            insertText: 'import pandas as pd',
            range: range
          },
          {
            label: 'from sklearn.model_selection import train_test_split',
            kind: monaco.languages.CompletionItemKind.Module,
            insertText: 'from sklearn.model_selection import train_test_split',
            range: range
          },
        ]

        return { suggestions }
      }
    })
  }

  // Execute code
  const handleExecute = async () => {
    setExecuting(true)
    setOutput('Executing...\n')

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/code/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: code,
          language: 'python',
          timeout: 5
        })
      })

      const result = await response.json()

      if (!response.ok) {
        setOutput(`ERROR: ${result.detail || 'Execution failed'}\n`)
        return
      }

      if (result.success) {
        let outputText = `Execution completed in ${result.execution_time}s\n\n`
        
        if (result.output) {
          outputText += 'Output:\n'
          outputText += result.output
        } else {
          outputText += '(No output)'
        }

        if (result.variables && Object.keys(result.variables).length > 0) {
          outputText += '\n\nVariables:\n'
          for (const [name, type] of Object.entries(result.variables)) {
            outputText += `  ${name}: ${type}\n`
          }
        }

        setOutput(outputText)
      } else {
        setOutput(`Execution failed:\n${result.error || result.stderr}`)
      }
    } catch (error) {
      setOutput(`Network error: ${error.message}\n\nMake sure the backend server is running.`)
    } finally {
      setExecuting(false)
    }
  }

  // Clear output
  const clearOutput = () => {
    setOutput('')
  }

  // Reset to default code
  const resetCode = () => {
    if (window.confirm('Reset to default code? This will clear your current work.')) {
      setCode(DEFAULT_CODE)
      setOutput('')
    }
  }

  return (
    <div className="code-editor-container">
      <div className="editor-toolbar">
        <button 
          className="toolbar-btn primary"
          onClick={handleExecute}
          disabled={executing}
        >
          {executing ? 'Running...' : 'Run Code (Ctrl+Enter)'}
        </button>
        <button className="toolbar-btn" onClick={clearOutput}>
          Clear Output
        </button>
        <button className="toolbar-btn" onClick={resetCode}>
          Reset Code
        </button>
        
        <div className="toolbar-spacer"></div>
        
        <label className="theme-selector">
          Theme:
          <select value={theme} onChange={e => setTheme(e.target.value)}>
            <option value="vs-dark">Dark</option>
            <option value="vs-light">Light</option>
            <option value="hc-black">High Contrast</option>
          </select>
        </label>

        {modelId && (
          <span className="model-indicator">
            Model: {modelId.substring(0, 8)}...
          </span>
        )}
      </div>

      <div className="editor-content">
        <Editor
          height="100%"
          language="python"
          theme={theme}
          value={code}
          onChange={setCode}
          onMount={handleEditorDidMount}
          options={editorOptions}
          loading={<div className="editor-loading">Loading Monaco Editor...</div>}
        />
      </div>

      <div className="output-panel">
        <div className="output-header">
          <h4>Output</h4>
          {output && (
            <span className="output-info">
              {output.includes('completed') ? 'Success' : output.includes('ERROR') || output.includes('failed') ? 'Error' : ''}
            </span>
          )}
        </div>
        <pre className={`output-content ${output.includes('ERROR') || output.includes('failed') ? 'error' : ''}`}>
          {output || '(Run code to see output)'}
        </pre>
      </div>
    </div>
  )
}
