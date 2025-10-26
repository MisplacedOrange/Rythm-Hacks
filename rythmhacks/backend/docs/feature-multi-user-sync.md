# Feature Documentation: Multi-User Real-Time Synchronization

## Overview
Real-time collaborative workspace enabling multiple users to simultaneously view training progress, edit code, adjust hyperparameters, and communicate. Built on WebSocket infrastructure with operational transformation for conflict resolution.

---

## Architecture Overview

### Communication Layers

1. **WebSocket Rooms**: User sessions grouped by workspace/experiment
2. **Presence System**: Track who's online and what they're viewing
3. **State Synchronization**: Broadcast state changes to all participants
4. **Conflict Resolution**: Handle simultaneous edits gracefully
5. **Permissions**: Role-based access control

---

## Backend Implementation

### WebSocket Room Manager

```python
# backend/app/core/collaboration.py
from fastapi import WebSocket
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio

@dataclass
class User:
    id: str
    name: str
    email: Optional[str]
    color: str  # For cursor/highlight color
    role: str  # 'owner', 'editor', 'viewer'

@dataclass
class UserPresence:
    user: User
    websocket: WebSocket
    workspace_id: str
    current_view: str  # 'dashboard', 'code_editor', 'data_table', etc.
    cursor_position: Optional[dict] = None
    last_active: datetime = datetime.utcnow()

class CollaborationManager:
    """Manage multi-user real-time collaboration"""
    
    def __init__(self):
        # workspace_id -> list of user presences
        self.active_users: Dict[str, List[UserPresence]] = {}
        
        # Track state for each workspace
        self.workspace_state: Dict[str, dict] = {}
        
        # Lock for state updates
        self.state_locks: Dict[str, asyncio.Lock] = {}
    
    async def connect_user(
        self, 
        websocket: WebSocket, 
        workspace_id: str, 
        user: User
    ):
        """Connect a user to a workspace"""
        await websocket.accept()
        
        # Initialize workspace if needed
        if workspace_id not in self.active_users:
            self.active_users[workspace_id] = []
            self.workspace_state[workspace_id] = {}
            self.state_locks[workspace_id] = asyncio.Lock()
        
        # Create presence
        presence = UserPresence(
            user=user,
            websocket=websocket,
            workspace_id=workspace_id,
            current_view='dashboard'
        )
        
        self.active_users[workspace_id].append(presence)
        
        # Notify others of new user
        await self.broadcast_to_workspace(workspace_id, {
            'type': 'user_joined',
            'user': {
                'id': user.id,
                'name': user.name,
                'color': user.color,
                'role': user.role
            },
            'timestamp': datetime.utcnow().isoformat()
        }, exclude=websocket)
        
        # Send current workspace state to new user
        await websocket.send_json({
            'type': 'workspace_state',
            'state': self.workspace_state[workspace_id],
            'users': [
                {
                    'id': p.user.id,
                    'name': p.user.name,
                    'color': p.user.color,
                    'current_view': p.current_view
                }
                for p in self.active_users[workspace_id]
            ]
        })
        
        return presence
    
    async def disconnect_user(self, presence: UserPresence):
        """Disconnect a user from workspace"""
        workspace_id = presence.workspace_id
        
        if workspace_id in self.active_users:
            self.active_users[workspace_id].remove(presence)
            
            # Notify others
            await self.broadcast_to_workspace(workspace_id, {
                'type': 'user_left',
                'user': {
                    'id': presence.user.id,
                    'name': presence.user.name
                },
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Clean up empty workspaces
            if not self.active_users[workspace_id]:
                del self.active_users[workspace_id]
                del self.workspace_state[workspace_id]
                del self.state_locks[workspace_id]
    
    async def broadcast_to_workspace(
        self, 
        workspace_id: str, 
        message: dict,
        exclude: Optional[WebSocket] = None
    ):
        """Send message to all users in a workspace"""
        if workspace_id not in self.active_users:
            return
        
        for presence in self.active_users[workspace_id]:
            if presence.websocket != exclude:
                try:
                    await presence.websocket.send_json(message)
                except:
                    # Connection lost, will be cleaned up
                    pass
    
    async def update_workspace_state(
        self, 
        workspace_id: str, 
        path: str, 
        value: any,
        user_id: str
    ):
        """
        Update workspace state with conflict resolution
        
        Path examples:
            - 'hyperparameters.learning_rate'
            - 'selected_algorithm'
            - 'training_status'
        """
        async with self.state_locks[workspace_id]:
            # Update state
            state = self.workspace_state[workspace_id]
            keys = path.split('.')
            
            # Navigate to nested key
            current = state
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Store old value for conflict detection
            old_value = current.get(keys[-1])
            
            # Update value
            current[keys[-1]] = value
            
            # Broadcast change
            await self.broadcast_to_workspace(workspace_id, {
                'type': 'state_update',
                'path': path,
                'value': value,
                'old_value': old_value,
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def update_user_presence(
        self,
        workspace_id: str,
        user_id: str,
        updates: dict
    ):
        """Update user presence information"""
        if workspace_id not in self.active_users:
            return
        
        for presence in self.active_users[workspace_id]:
            if presence.user.id == user_id:
                # Update fields
                if 'current_view' in updates:
                    presence.current_view = updates['current_view']
                if 'cursor_position' in updates:
                    presence.cursor_position = updates['cursor_position']
                
                presence.last_active = datetime.utcnow()
                
                # Broadcast presence update
                await self.broadcast_to_workspace(workspace_id, {
                    'type': 'presence_update',
                    'user_id': user_id,
                    'updates': updates,
                    'timestamp': datetime.utcnow().isoformat()
                })
                break
    
    def get_workspace_users(self, workspace_id: str) -> List[dict]:
        """Get list of users in a workspace"""
        if workspace_id not in self.active_users:
            return []
        
        return [
            {
                'id': p.user.id,
                'name': p.user.name,
                'email': p.user.email,
                'color': p.user.color,
                'role': p.user.role,
                'current_view': p.current_view,
                'last_active': p.last_active.isoformat()
            }
            for p in self.active_users[workspace_id]
        ]

# Global instance
collaboration_manager = CollaborationManager()
```

---

### WebSocket Endpoint

```python
# backend/app/api/websockets/collaboration.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from backend.app.core.collaboration import collaboration_manager, User
import json

router = APIRouter()

@router.websocket("/ws/workspace/{workspace_id}")
async def workspace_websocket(
    websocket: WebSocket,
    workspace_id: str,
    user_id: str,
    user_name: str,
    user_role: str = "editor"
):
    """
    WebSocket endpoint for workspace collaboration
    
    Query params:
        user_id: Unique user identifier
        user_name: Display name
        user_role: 'owner', 'editor', or 'viewer'
    """
    # Generate user color
    user_color = generate_user_color(user_id)
    
    user = User(
        id=user_id,
        name=user_name,
        email=None,  # Would come from auth
        color=user_color,
        role=user_role
    )
    
    presence = await collaboration_manager.connect_user(
        websocket, workspace_id, user
    )
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get('type')
            
            # Handle different message types
            if message_type == 'state_update':
                # Update workspace state
                await collaboration_manager.update_workspace_state(
                    workspace_id,
                    message['path'],
                    message['value'],
                    user_id
                )
            
            elif message_type == 'presence_update':
                # Update user presence
                await collaboration_manager.update_user_presence(
                    workspace_id,
                    user_id,
                    message['updates']
                )
            
            elif message_type == 'cursor_position':
                # Broadcast cursor position for collaborative editing
                await collaboration_manager.broadcast_to_workspace(
                    workspace_id,
                    {
                        'type': 'cursor_position',
                        'user_id': user_id,
                        'user_name': user_name,
                        'user_color': user_color,
                        'position': message['position'],
                        'view': message.get('view', 'code_editor')
                    },
                    exclude=websocket
                )
            
            elif message_type == 'ping':
                # Heartbeat
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        await collaboration_manager.disconnect_user(presence)

def generate_user_color(user_id: str) -> str:
    """Generate consistent color for user"""
    colors = [
        '#F4A460', '#4682B4', '#5B9BD5', '#28a745',
        '#dc3545', '#ffc107', '#17a2b8', '#6c757d',
        '#e83e8c', '#fd7e14', '#20c997', '#6610f2'
    ]
    # Hash user_id to pick color
    hash_val = sum(ord(c) for c in user_id)
    return colors[hash_val % len(colors)]
```

---

### REST API Endpoints

```python
# backend/app/api/routes/workspace.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

@router.get("/api/workspaces/{workspace_id}/users")
async def get_workspace_users(workspace_id: str):
    """Get list of currently active users in workspace"""
    users = collaboration_manager.get_workspace_users(workspace_id)
    return {'users': users}

@router.get("/api/workspaces/{workspace_id}/state")
async def get_workspace_state(workspace_id: str):
    """Get current workspace state"""
    if workspace_id not in collaboration_manager.workspace_state:
        raise HTTPException(404, "Workspace not found")
    
    return {
        'state': collaboration_manager.workspace_state[workspace_id],
        'active_users': len(collaboration_manager.active_users.get(workspace_id, []))
    }

class InviteRequest(BaseModel):
    email: str
    role: str  # 'editor' or 'viewer'

@router.post("/api/workspaces/{workspace_id}/invite")
async def invite_user(workspace_id: str, request: InviteRequest):
    """Invite user to workspace"""
    # Generate invitation token
    invite_token = generate_invite_token(workspace_id, request.email, request.role)
    
    # Send email invitation
    await send_invitation_email(request.email, workspace_id, invite_token)
    
    return {
        'invite_token': invite_token,
        'invite_url': f'/workspace/{workspace_id}/join?token={invite_token}'
    }

@router.post("/api/workspaces/{workspace_id}/join")
async def join_workspace(workspace_id: str, token: str):
    """Join workspace via invitation token"""
    # Validate token
    invitation = validate_invite_token(token)
    
    if not invitation:
        raise HTTPException(400, "Invalid invitation token")
    
    # Add user to workspace permissions
    await add_workspace_permission(
        workspace_id,
        invitation['user_email'],
        invitation['role']
    )
    
    return {'status': 'joined', 'role': invitation['role']}
```

---

## Frontend Implementation

### Collaboration Provider

```javascript
// frontend/src/contexts/CollaborationContext.jsx
import React, { createContext, useContext, useState, useEffect, useRef } from 'react'

const CollaborationContext = createContext()

export function CollaborationProvider({ workspaceId, userId, userName, children }) {
  const [users, setUsers] = useState([])
  const [workspaceState, setWorkspaceState] = useState({})
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket(
      `ws://localhost:8000/ws/workspace/${workspaceId}?` +
      `user_id=${userId}&user_name=${encodeURIComponent(userName)}&user_role=editor`
    )

    ws.onopen = () => {
      console.log('Connected to workspace')
      setConnected(true)
    }

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)
      handleMessage(message)
    }

    ws.onclose = () => {
      console.log('Disconnected from workspace')
      setConnected(false)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    wsRef.current = ws

    // Heartbeat
    const heartbeat = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 30000)

    return () => {
      clearInterval(heartbeat)
      if (ws) ws.close()
    }
  }, [workspaceId, userId, userName])

  const handleMessage = (message) => {
    switch (message.type) {
      case 'workspace_state':
        setWorkspaceState(message.state)
        setUsers(message.users)
        break

      case 'user_joined':
        setUsers(prev => [...prev, message.user])
        showNotification(`${message.user.name} joined the workspace`)
        break

      case 'user_left':
        setUsers(prev => prev.filter(u => u.id !== message.user.id))
        showNotification(`${message.user.name} left the workspace`)
        break

      case 'state_update':
        updateWorkspaceState(message.path, message.value)
        break

      case 'presence_update':
        updateUserPresence(message.user_id, message.updates)
        break

      case 'cursor_position':
        updateUserCursor(message)
        break
    }
  }

  const updateWorkspaceState = (path, value) => {
    setWorkspaceState(prev => {
      const newState = { ...prev }
      const keys = path.split('.')
      let current = newState

      for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) current[keys[i]] = {}
        current = current[keys[i]]
      }

      current[keys[keys.length - 1]] = value
      return newState
    })
  }

  const updateUserPresence = (userId, updates) => {
    setUsers(prev => prev.map(user => 
      user.id === userId ? { ...user, ...updates } : user
    ))
  }

  const updateUserCursor = (message) => {
    // Dispatch custom event for cursor visualization
    window.dispatchEvent(new CustomEvent('remote-cursor', {
      detail: message
    }))
  }

  const sendStateUpdate = (path, value) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'state_update',
        path,
        value
      }))
    }
  }

  const sendPresenceUpdate = (updates) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'presence_update',
        updates
      }))
    }
  }

  const value = {
    users,
    workspaceState,
    connected,
    sendStateUpdate,
    sendPresenceUpdate
  }

  return (
    <CollaborationContext.Provider value={value}>
      {children}
    </CollaborationContext.Provider>
  )
}

export const useCollaboration = () => useContext(CollaborationContext)

const showNotification = (message) => {
  // Browser notification or toast
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification('MediLytica', { body: message })
  }
}
```

---

### User Avatars Component

```javascript
// frontend/src/components/ActiveUsers.jsx
import React from 'react'
import { useCollaboration } from '../contexts/CollaborationContext'
import './ActiveUsers.css'

export default function ActiveUsers() {
  const { users, connected } = useCollaboration()

  return (
    <div className="active-users">
      <div className="connection-status">
        <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
        {connected ? 'Live' : 'Offline'}
      </div>

      <div className="users-list">
        {users.map(user => (
          <div
            key={user.id}
            className="user-avatar"
            style={{ borderColor: user.color }}
            title={`${user.name} - ${user.role} - ${user.current_view || 'dashboard'}`}
          >
            {user.name.charAt(0).toUpperCase()}
          </div>
        ))}
      </div>

      <div className="user-count">
        {users.length} {users.length === 1 ? 'user' : 'users'} online
      </div>
    </div>
  )
}
```

---

### Synchronized Hyperparameters

```javascript
// Update Dashboard.jsx to sync hyperparameters
import { useCollaboration } from '../contexts/CollaborationContext'

export default function Dashboard() {
  const { workspaceState, sendStateUpdate } = useCollaboration()
  
  const handleHyperparameterChange = (param, value) => {
    // Update local state
    setHyperparameters(prev => ({ ...prev, [param]: value }))
    
    // Broadcast to other users
    sendStateUpdate(`hyperparameters.${param}`, value)
  }

  // Listen for remote updates
  useEffect(() => {
    if (workspaceState.hyperparameters) {
      setHyperparameters(workspaceState.hyperparameters)
    }
  }, [workspaceState.hyperparameters])

  // ... rest of component
}
```

---

## Conflict Resolution

### Operational Transformation for Text

```python
# backend/app/core/ot.py
from typing import List, Tuple

class Operation:
    """Operational Transformation operation"""
    
    def __init__(self, op_type: str, position: int, data: str = None):
        self.type = op_type  # 'insert', 'delete', 'retain'
        self.position = position
        self.data = data

def transform_operations(op1: Operation, op2: Operation) -> Tuple[Operation, Operation]:
    """
    Transform two concurrent operations
    
    Returns transformed versions that can be applied in any order
    """
    # Insert vs Insert
    if op1.type == 'insert' and op2.type == 'insert':
        if op1.position < op2.position:
            return op1, Operation('insert', op2.position + len(op1.data), op2.data)
        elif op1.position > op2.position:
            return Operation('insert', op1.position + len(op2.data), op1.data), op2
        else:
            # Same position - use tie-breaker (e.g., user_id)
            return op1, Operation('insert', op2.position + len(op1.data), op2.data)
    
    # Insert vs Delete
    elif op1.type == 'insert' and op2.type == 'delete':
        if op1.position <= op2.position:
            return op1, Operation('delete', op2.position + len(op1.data))
        else:
            return Operation('insert', op1.position - 1, op1.data), op2
    
    # Delete vs Insert
    elif op1.type == 'delete' and op2.type == 'insert':
        if op2.position <= op1.position:
            return Operation('delete', op1.position + len(op2.data)), op2
        else:
            return op1, Operation('insert', op2.position - 1, op2.data)
    
    # Delete vs Delete
    elif op1.type == 'delete' and op2.type == 'delete':
        if op1.position < op2.position:
            return op1, Operation('delete', op2.position - 1)
        elif op1.position > op2.position:
            return Operation('delete', op1.position - 1), op2
        else:
            # Same position - one wins
            return op1, None
    
    return op1, op2
```

---

## Styling

```css
/* frontend/src/components/ActiveUsers.css */
.active-users {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 16px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #666;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.status-dot.connected {
  background: #28a745;
}

.status-dot.disconnected {
  background: #dc3545;
  animation: none;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.users-list {
  display: flex;
  gap: 8px;
}

.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #4682B4;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 14px;
  border: 2px solid;
  cursor: pointer;
  transition: transform 0.2s;
}

.user-avatar:hover {
  transform: scale(1.1);
}

.user-count {
  font-size: 12px;
  color: #999;
  margin-left: auto;
}
```

---

## Security & Permissions

### Role-Based Access Control

```python
# backend/app/core/permissions.py
from enum import Enum

class Role(str, Enum):
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"

class Permission(str, Enum):
    VIEW_WORKSPACE = "view_workspace"
    EDIT_HYPERPARAMETERS = "edit_hyperparameters"
    START_TRAINING = "start_training"
    EDIT_CODE = "edit_code"
    UPLOAD_DATA = "upload_data"
    INVITE_USERS = "invite_users"
    MANAGE_PERMISSIONS = "manage_permissions"

ROLE_PERMISSIONS = {
    Role.OWNER: list(Permission),
    Role.EDITOR: [
        Permission.VIEW_WORKSPACE,
        Permission.EDIT_HYPERPARAMETERS,
        Permission.START_TRAINING,
        Permission.EDIT_CODE,
        Permission.UPLOAD_DATA
    ],
    Role.VIEWER: [
        Permission.VIEW_WORKSPACE
    ]
}

def has_permission(role: Role, permission: Permission) -> bool:
    """Check if role has permission"""
    return permission in ROLE_PERMISSIONS.get(role, [])
```

---

## Future Enhancements

1. **Video/Voice Chat**: Integrate WebRTC for calls
2. **Screen Sharing**: Share workspace view
3. **Annotation Tools**: Draw/highlight on visualizations
4. **Activity Feed**: Timeline of all workspace changes
5. **Offline Mode**: Queue changes when disconnected
6. **Conflict Alerts**: Notify users of conflicts
7. **Version History**: Time-travel through workspace states
8. **Cursor Following**: Follow another user's cursor
9. **Collaborative Debugging**: Share breakpoints
10. **Export Session**: Download collaboration transcript
