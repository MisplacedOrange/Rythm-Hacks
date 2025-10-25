import React, { useState } from 'react'
import './Chat.css'

const Chat = () => {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'James', text: 'What problems are there with our current model?' },
    { id: 2, sender: 'Maria', text: "I looked at the model accuracy compared to the training. It seems like it's overfitting." },
    { id: 3, sender: 'Me', text: 'What do you guys think about this data?' },
    { id: 4, sender: 'James', text: 'I think its great!' },
    { id: 5, sender: 'Me', text: 'I agree!' },
    { id: 6, sender: 'James', text: 'We should definitely talk more about this.' },
    { id: 7, sender: 'Me', text: 'Definitely!' },
  ])
  const [inputValue, setInputValue] = useState('')

  const handleSend = () => {
    if (inputValue.trim()) {
      setMessages([...messages, { id: messages.length + 1, sender: 'Me', text: inputValue }])
      setInputValue('')
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h3>Chat</h3>
      </div>
      
      <div className="chat-toolbar">
        <button className="toolbar-btn" title="Bold"><strong>B</strong></button>
        <button className="toolbar-btn" title="Italic"><em>I</em></button>
        <button className="toolbar-btn" title="Underline"><u>U</u></button>
        <button className="toolbar-btn" title="Strikethrough"><s>A</s></button>
        <button className="toolbar-btn" title="Link">🔗</button>
        <button className="toolbar-btn" title="Code">{'{ }'}</button>
        <button className="toolbar-btn" title="Image">📷</button>
        <button className="toolbar-btn" title="List">☰</button>
        <button className="toolbar-btn" title="Emoji">😊</button>
        <button className="toolbar-btn" title="Bullet List">•</button>
        <button className="toolbar-btn" title="Numbered List">1.</button>
        <button className="toolbar-btn" title="Quote">"</button>
        <button className="toolbar-btn" title="Outdent">«</button>
        <button className="toolbar-btn" title="Indent">»</button>
        <button className="toolbar-btn" title="Clear">×</button>
      </div>

      <div className="chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.sender === 'Me' ? 'me' : ''}`}>
            <span className="message-sender">{msg.sender}:</span> {msg.text}
          </div>
        ))}
      </div>

      <div className="chat-input">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type a message..."
          rows="3"
        />
        <button className="send-btn" onClick={handleSend}>Send</button>
      </div>
    </div>
  )
}

export default Chat
