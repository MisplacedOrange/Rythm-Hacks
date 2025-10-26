# Feature Documentation: Team Collaboration Chat

## Overview
Real-time team chat interface with rich text formatting toolbar, enabling ML practitioners to discuss models, share observations, and collaborate on experiments within the same workspace.

---

## Current Implementation

### Frontend Component

**Location**: `components/Chat.jsx`

**State Management**:
```javascript
const [messages, setMessages] = useState([
  { id: 1, sender: 'James', text: 'What problems are there with our current model?' },
  { id: 2, sender: 'Maria', text: "I looked at the model accuracy compared to the training. It seems like it's overfitting." },
  //... more messages
])
const [inputValue, setInputValue] = useState('')
```

**Features**:
- Rich text toolbar (Bold, Italic, Underline, Links, Code, Images, Lists)
- Message history display
- Differentiated styling for "Me" vs other users
- Text input with Send button
- Enter key to send, Shift+Enter for new line

---

## Backend Integration

### WebSocket Server

**WebSocket Endpoint**: `WS /ws/chat/{room_id}`

```python
# backend/app/api/websockets/chat.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
from datetime import datetime

class ConnectionManager:
    """Manage WebSocket connections for chat rooms"""
    
    def __init__(self):
        # room_id -> list of websockets
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # room_id -> list of users
        self.room_users: Dict[str, List[dict]] = {}
    
    async def connect(self, websocket: WebSocket, room_id: str, user: dict):
        """Add connection to room"""
        await websocket.accept()
        
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
            self.room_users[room_id] = []
        
        self.active_connections[room_id].append(websocket)
        self.room_users[room_id].append(user)
        
        # Notify room of new user
        await self.broadcast_to_room(room_id, {
            'type': 'user_joined',
            'user': user,
            'users_in_room': self.room_users[room_id]
        })
    
    def disconnect(self, websocket: WebSocket, room_id: str):
        """Remove connection from room"""
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            
            # Clean up empty rooms
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
                del self.room_users[room_id]
    
    async def broadcast_to_room(self, room_id: str, message: dict):
        """Send message to all connections in a room"""
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                try:
                    await connection.send_json(message)
                except:
                    # Connection lost, will be cleaned up on disconnect
                    pass
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        await websocket.send_json(message)

manager = ConnectionManager()

@router.websocket("/ws/chat/{room_id}")
async def chat_websocket(websocket: WebSocket, room_id: str, user_id: str = None):
    """
    WebSocket endpoint for chat
    
    Query params:
        user_id: Identifier for the user (temporary, until auth implemented)
    """
    # Get or create user info
    user = {
        'id': user_id or generate_temp_user_id(),
        'name': get_user_name(user_id) if user_id else 'Anonymous',
        'joined_at': datetime.utcnow().isoformat()
    }
    
    await manager.connect(websocket, room_id, user)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message based on type
            if message_data['type'] == 'chat_message':
                # Save to database
                saved_message = await save_message(room_id, user, message_data)
                
                # Broadcast to room
                await manager.broadcast_to_room(room_id, {
                    'type': 'new_message',
                    'message': saved_message
                })
            
            elif message_data['type'] == 'typing':
                # Notify others that user is typing
                await manager.broadcast_to_room(room_id, {
                    'type': 'user_typing',
                    'user': user['name']
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        await manager.broadcast_to_room(room_id, {
            'type': 'user_left',
            'user': user
        })
```

---

### Message Storage

#### Database Schema

```python
# backend/app/models/chat.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

class ChatRoom(Base):
    __tablename__ = "chat_rooms"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    experiment_id = Column(String, ForeignKey("experiments.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    messages = relationship("ChatMessage", back_populates="room")
    participants = relationship("ChatParticipant", back_populates="room")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True)
    room_id = Column(String, ForeignKey("chat_rooms.id"))
    user_id = Column(String, nullable=True)  # For future auth
    user_name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String, default='text')  # text, code, image
    formatting = Column(JSON, nullable=True)  # Rich text formatting
    timestamp = Column(DateTime, default=datetime.utcnow)
    edited_at = Column(DateTime, nullable=True)
    
    room = relationship("ChatRoom", back_populates="messages")

class ChatParticipant(Base):
    __tablename__ = "chat_participants"
    
    id = Column(String, primary_key=True)
    room_id = Column(String, ForeignKey("chat_rooms.id"))
    user_id = Column(String, nullable=True)
    user_name = Column(String, nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    room = relationship("ChatRoom", back_populates="participants")
```

---

### REST API Endpoints

```python
# backend/app/api/routes/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class CreateRoomRequest(BaseModel):
    name: str
    experiment_id: Optional[str] = None

class SendMessageRequest(BaseModel):
    content: str
    content_type: str = 'text'
    formatting: Optional[dict] = None

@router.post("/api/chat/rooms")
async def create_room(request: CreateRoomRequest):
    """Create a new chat room"""
    room = ChatRoom(
        id=generate_unique_id(),
        name=request.name,
        experiment_id=request.experiment_id
    )
    
    db.add(room)
    db.commit()
    
    return {
        'room_id': room.id,
        'name': room.name,
        'websocket_url': f'/ws/chat/{room.id}'
    }

@router.get("/api/chat/rooms/{room_id}/messages")
async def get_messages(
    room_id: str,
    limit: int = 50,
    before: Optional[str] = None
):
    """
    Get message history for a room
    
    Args:
        room_id: Chat room ID
        limit: Maximum messages to return
        before: Get messages before this message ID (for pagination)
    """
    query = db.query(ChatMessage).filter(ChatMessage.room_id == room_id)
    
    if before:
        before_msg = db.query(ChatMessage).get(before)
        query = query.filter(ChatMessage.timestamp < before_msg.timestamp)
    
    messages = query.order_by(ChatMessage.timestamp.desc()).limit(limit).all()
    
    return {
        'messages': [
            {
                'id': msg.id,
                'user_name': msg.user_name,
                'content': msg.content,
                'content_type': msg.content_type,
                'formatting': msg.formatting,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in reversed(messages)
        ]
    }

@router.post("/api/chat/rooms/{room_id}/messages")
async def send_message(room_id: str, request: SendMessageRequest, user_id: str = None):
    """
    Send a message (alternative to WebSocket for reliability)
    """
    message = ChatMessage(
        id=generate_unique_id(),
        room_id=room_id,
        user_id=user_id,
        user_name=get_user_name(user_id) if user_id else 'Anonymous',
        content=request.content,
        content_type=request.content_type,
        formatting=request.formatting
    )
    
    db.add(message)
    db.commit()
    
    # Broadcast via WebSocket
    await manager.broadcast_to_room(room_id, {
        'type': 'new_message',
        'message': {
            'id': message.id,
            'user_name': message.user_name,
            'content': message.content,
            'timestamp': message.timestamp.isoformat()
        }
    })
    
    return {'message_id': message.id}

@router.get("/api/chat/rooms/{room_id}/participants")
async def get_participants(room_id: str):
    """Get current active participants in a room"""
    participants = db.query(ChatParticipant).filter(
        ChatParticipant.room_id == room_id
    ).all()
    
    return {
        'participants': [
            {
                'user_id': p.user_id,
                'user_name': p.user_name,
                'joined_at': p.joined_at.isoformat(),
                'is_online': is_user_online(p.user_id, room_id)
            }
            for p in participants
        ]
    }
```

---

## Enhanced Frontend Implementation

### WebSocket Integration

```javascript
// frontend/src/components/Chat.jsx
import React, { useState, useEffect, useRef } from 'react'
import './Chat.css'

export default function Chat({ roomId, userId, userName }) {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [ws, setWs] = useState(null)
  const [connected, setConnected] = useState(false)
  const [participants, setParticipants] = useState([])
  const messagesEndRef = useRef(null)

  // Connect to WebSocket
  useEffect(() => {
    const websocket = new WebSocket(
      `ws://localhost:8000/ws/chat/${roomId}?user_id=${userId}`
    )
    
    websocket.onopen = () => {
      console.log('Connected to chat')
      setConnected(true)
      loadMessageHistory()
    }
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      handleWebSocketMessage(data)
    }
    
    websocket.onclose = () => {
      console.log('Disconnected from chat')
      setConnected(false)
    }
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    setWs(websocket)
    
    return () => {
      if (websocket) {
        websocket.close()
      }
    }
  }, [roomId, userId])

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'new_message':
        setMessages(prev => [...prev, data.message])
        scrollToBottom()
        break
      
      case 'user_joined':
        setParticipants(data.users_in_room)
        addSystemMessage(`${data.user.name} joined the chat`)
        break
      
      case 'user_left':
        addSystemMessage(`${data.user.name} left the chat`)
        break
      
      case 'user_typing':
        showTypingIndicator(data.user)
        break
    }
  }

  // Load message history
  const loadMessageHistory = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/chat/rooms/${roomId}/messages?limit=50`
      )
      const data = await response.json()
      setMessages(data.messages)
      scrollToBottom()
    } catch (error) {
      console.error('Failed to load messages:', error)
    }
  }

  // Send message
  const handleSend = () => {
    if (!inputValue.trim() || !ws || !connected) return
    
    const message = {
      type: 'chat_message',
      content: inputValue,
      content_type: 'text',
      user_name: userName
    }
    
    ws.send(JSON.stringify(message))
    setInputValue('')
  }

  // Typing indicator
  const handleTyping = () => {
    if (ws && connected) {
      ws.send(JSON.stringify({ type: 'typing' }))
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h3>Chat</h3>
        <div className="connection-status">
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
        <div className="participants-count">
          {participants.length} online
        </div>
      </div>
      
      <div className="chat-toolbar">
        {/* Rich text toolbar buttons */}
      </div>

      <div className="chat-messages">
        {messages.map((msg) => (
          <div 
            key={msg.id} 
            className={`chat-message ${msg.user_name === userName ? 'me' : ''}`}
          >
            <span className="message-sender">{msg.user_name}:</span> {msg.content}
            <span className="message-time">
              {new Date(msg.timestamp).toLocaleTimeString()}
            </span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <textarea
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value)
            handleTyping()
          }}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSend()
            }
          }}
          placeholder="Type a message..."
          rows="3"
          disabled={!connected}
        />
        <button 
          className="send-btn" 
          onClick={handleSend}
          disabled={!connected || !inputValue.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}
```

---

## Rich Text Formatting

### Frontend Formatting State

```javascript
const [formatting, setFormatting] = useState({
  bold: false,
  italic: false,
  underline: false,
  code: false
})

const applyFormatting = (type) => {
  setFormatting(prev => ({
    ...prev,
    [type]: !prev[type]
  }))
}

const formatMessage = (text, formatting) => {
  let formatted = text
  
  if (formatting.bold) formatted = `**${formatted}**`
  if (formatting.italic) formatted = `*${formatted}*`
  if (formatting.code) formatted = `\`${formatted}\``
  
  return formatted
}
```

### Backend Markdown Processing

```python
# backend/app/utils/markdown.py
import markdown
from bleach import clean

ALLOWED_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'code', 'pre',
    'a', 'ul', 'ol', 'li', 'blockquote'
]

def process_message_content(content: str, content_type: str = 'text'):
    """Process and sanitize message content"""
    if content_type == 'markdown':
        # Convert markdown to HTML
        html = markdown.markdown(content)
        # Sanitize to prevent XSS
        clean_html = clean(html, tags=ALLOWED_TAGS)
        return clean_html
    elif content_type == 'code':
        # Syntax highlighting for code blocks
        return f'<pre><code>{content}</code></pre>'
    else:
        # Plain text - escape HTML
        return clean(content, tags=[], strip=True)
```

---

## Notifications & Mentions

### @Mention Feature

```python
@router.post("/api/chat/rooms/{room_id}/messages")
async def send_message(room_id: str, request: SendMessageRequest, user_id: str = None):
    # ... existing code ...
    
    # Extract mentions
    mentions = extract_mentions(request.content)
    
    # Notify mentioned users
    for mentioned_user in mentions:
        await notify_user(mentioned_user, {
            'type': 'mention',
            'room_id': room_id,
            'message_id': message.id,
            'from_user': user_name
        })
```

---

## Future Enhancements

1. **File Attachments**: Share images, CSVs, model files
2. **Code Snippets**: Syntax-highlighted code blocks
3. **Reactions**: Emoji reactions to messages
4. **Threads**: Threaded conversations
5. **Search**: Full-text search through chat history
6. **Notifications**: Desktop/email notifications for mentions
7. **Voice/Video**: Audio/video calls integration
8. **Message Editing**: Edit sent messages
9. **Read Receipts**: See who has read messages
10. **Bot Integration**: AI assistant for answering questions
## MVP Additions (Missing)

- Message schema: { id, roomId, userId, text, createdAt }
- Local persistence (per route/dataset) before backend.
- Basic actions: send, delete own message, copy link.
- Presence/typing deferred; no realtime in MVP.
## Integration with Code Editor Collaboration

- Shared identity/presence model: same `roomId` scheme (route + datasetId).
- Presence events reused: `presence` updates set status for both chat and editor.
- Linking: messages may reference editor selections or files with `ref: {type:'editor', range}`.

## Backend/Frontend Techstack Interactions

- Backend: same WS server can multiplex channels (`type: 'chat' | 'editor'`).
- Frontend: a single WS connection per room; message router dispatches to Chat or Editor handlers.
- Persistence: chat messages appended to `{roomId}.jsonl`; editor file stored separately.
