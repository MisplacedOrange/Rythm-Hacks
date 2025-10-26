// Simple Ably client wrapper with graceful no-key fallback
// Usage: import { getRoomChannel, getClientId } from './realtime/ablyClient'


let ablyRealtime = null
let ablyAvailable = false
let clientId = null

function ensureClientId() {
  if (!clientId) {
    try {
      const stored = localStorage.getItem('rh_client_id')
      clientId = stored || `user_${Math.random().toString(36).slice(2, 8)}`
      if (!stored) localStorage.setItem('rh_client_id', clientId)
    } catch {
      clientId = `user_${Math.random().toString(36).slice(2, 8)}`
    }
  }
  return clientId
}

export function getClientId() {
  return ensureClientId()
}

async function initAbly() {
  if (ablyRealtime || ablyAvailable) return { ablyAvailable, ablyRealtime }
  console.log('🔍 All env vars:', import.meta.env)
  const key = import.meta?.env?.VITE_ABLY_KEY
  console.log('🔑 Ably Key Check:', key ? '✅ Key Found' : '❌ No Key Found')
  console.log('🔑 Raw key value:', key)
  if (!key) {
    console.warn('⚠️ No Ably key - using BroadcastChannel fallback (same-browser only)')
    return { ablyAvailable: false, ablyRealtime: null }
  }
  try {
    console.log('⚙️ Importing Ably SDK...')
    const Ably = await import('ably')
    const clientId = ensureClientId()
    console.log('👤 Client ID:', clientId)
    ablyRealtime = new Ably.Realtime({ key, clientId })
    ablyAvailable = true
    console.log('✅ Ably initialized successfully!')
    
    // Listen for connection state changes
    ablyRealtime.connection.on('connected', () => {
      console.log('🟢 Ably connected!')
    })
    ablyRealtime.connection.on('failed', (err) => {
      console.error('🔴 Ably connection failed:', err)
    })
  } catch (e) {
    console.warn('❌ Ably import failed; falling back to BroadcastChannel:', e)
    ablyAvailable = false
  }
  return { ablyAvailable, ablyRealtime }
}

// Fallback BroadcastChannel for same-tab demos (not cross-origin)
const channels = {}
function getBroadcastChannel(name) {
  if (!channels[name]) {
    try {
      channels[name] = new BroadcastChannel(name)
    } catch {
      channels[name] = {
        postMessage: () => {},
        addEventListener: () => {},
        removeEventListener: () => {},
        close: () => {},
      }
    }
  }
  return channels[name]
}

export async function createRoomChannel(roomId, sub = 'main') {
  const { ablyAvailable, ablyRealtime } = await initAbly()
  const channelName = `room:${roomId}:${sub}`
  console.log(`📡 Creating channel: ${channelName}`)

  if (ablyAvailable && ablyRealtime) {
    console.log('✅ Using Ably for real-time sync')
    const ch = ablyRealtime.channels.get(channelName)
    return {
      type: 'ably',
      subscribe: (event, handler) => {
        console.log(`👂 Subscribing to event: ${event}`)
        ch.subscribe(event, (msg) => {
          console.log(`📨 Received message on ${event}:`, msg.data)
          handler(msg.data, msg)
        })
      },
      publish: (event, data) => {
        console.log(`📤 Publishing to ${event}:`, data)
        ch.publish(event, data)
      },
      presenceEnter: async (data) => ch.presence.enter(data),
      presenceLeave: async () => ch.presence.leave(),
      presenceGet: async () => ch.presence.get(),
      close: () => ch.detach(),
    }
  }

  // BroadcastChannel fallback
  console.log('⚠️ Using BroadcastChannel fallback (same-browser only)')
  const bc = getBroadcastChannel(channelName)
  const listeners = new Map()
  const onMessage = (e) => {
    const { event, data } = e.data || {}
    const handler = listeners.get(event)
    if (handler) handler(data, { local: true })
  }
  bc.addEventListener?.('message', onMessage)
  return {
    type: 'broadcast',
    subscribe: (event, handler) => listeners.set(event, handler),
    publish: (event, data) => bc.postMessage({ event, data }),
    presenceEnter: async () => {},
    presenceLeave: async () => {},
    presenceGet: async () => [],
    close: () => bc.removeEventListener?.('message', onMessage),
  }
}

export function newRoomId() {
  return Math.random().toString(36).slice(2, 10)
}
