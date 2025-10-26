import React, { useEffect, useRef, useState } from 'react'
import './Chat.css'
import useRoom from '../hooks/useRoom'
import { getClientId } from '../realtime/ablyClient'

const Chat = () => {
  const { publish, subscribe, shareUrl, roomId } = useRoom('chat')
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const selfId = getClientId()
  const seenIds = useRef(new Set())
  const responseIndex = useRef(0)

  const copyShareUrl = () => {
    navigator.clipboard.writeText(shareUrl)
    alert('Share URL copied! Send this to collaborate.')
  }

  // ML-related auto-responses from collaborator (in order)
  const mlResponses = [
    "Great idea! Have you considered trying a random forest for better accuracy?",
    "That makes sense. We should also check for overfitting with cross-validation.",
    "Good point. Let's tune the hyperparameters using grid search.",
    "I agree. Maybe we should normalize the features first before training.",
    "Interesting! What about adding dropout layers to prevent overfitting?",
    "True. We could also try ensemble methods to improve the predictions.",
    "Makes sense. Let's visualize the confusion matrix to see where it's failing.",
    "Nice! We should also check the learning curves to see if we need more data.",
    "Exactly. Have you tried using SMOTE to handle the class imbalance?",
    "Good thinking. We could increase the batch size to speed up training.",
    "Right. Let's compare this with a gradient boosting model too.",
    "Absolutely. We should track the loss function across epochs.",
    "I see. Maybe we need to adjust the learning rate for better convergence.",
    "That works. Let's also save checkpoints during training.",
    "Perfect. We should test this on the validation set before deploying."
  ]

  const getNextResponse = () => {
    const response = mlResponses[responseIndex.current]
    responseIndex.current = (responseIndex.current + 1) % mlResponses.length
    return response
  }

  useEffect(() => {
    const handler = (data) => {
      if (!data || !data.id || seenIds.current.has(data.id)) return
      seenIds.current.add(data.id)
      setMessages((prev) => [...prev, { id: data.id, sender: data.sender, text: data.text }])
    }
    subscribe('chat:message', handler)
  }, [subscribe])

  const handleSend = () => {
    if (inputValue.trim()) {
      const msg = { id: `${Date.now()}_${Math.random().toString(36).slice(2,6)}`, sender: selfId, text: inputValue }
      publish('chat:message', msg)
      // Optimistic update for instant UX; de-duped by id on receive
      setMessages((prev) => [...prev, msg])
      setInputValue('')

      // Auto-respond from "collaborator" after 1-2 seconds
      setTimeout(() => {
        const responseMsg = {
          id: `${Date.now()}_${Math.random().toString(36).slice(2,6)}`,
          sender: 'Collaborator',
          text: getNextResponse()
        }
        setMessages((prev) => [...prev, responseMsg])
      }, 1000 + Math.random() * 1000) // Random delay between 1-2 seconds
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
        <button 
          className="share-room-btn" 
          onClick={copyShareUrl}
          title="Copy share URL to collaborate"
        >
          🔗 Share Room
        </button>
      </div>
      
      <div className="room-info">
        <small>Room: {roomId} • Client: {selfId}</small>
      </div>

      <div className="chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`chat-message ${msg.sender === selfId ? 'me' : ''}`}>
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
