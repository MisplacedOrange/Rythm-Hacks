import { useEffect, useMemo, useRef, useState } from 'react'
import { createRoomChannel, getClientId, newRoomId } from '../realtime/ablyClient'

export function getOrCreateRoomFromUrl() {
  const url = new URL(typeof window !== 'undefined' ? window.location.href : 'http://localhost')
  let room = url.searchParams.get('room')
  if (!room) {
    room = newRoomId()
    try {
      url.searchParams.set('room', room)
      window.history.replaceState({}, '', url.toString())
    } catch {}
  }
  return room
}

export default function useRoom(sub = 'main') {
  const [roomId] = useState(() => getOrCreateRoomFromUrl())
  const channelRef = useRef(null)
  const [users, setUsers] = useState([])

  useEffect(() => {
    let ch
    let mounted = true
    ;(async () => {
      ch = await createRoomChannel(roomId, sub)
      if (!mounted) return
      channelRef.current = ch
      try {
        await ch.presenceEnter({ id: getClientId(), ts: Date.now() })
        const current = await ch.presenceGet()
        setUsers(Array.isArray(current) ? current.map((x) => x.clientId || x.id || 'user') : [])
      } catch {}
    })()
    return () => {
      mounted = false
      if (ch) {
        try { ch.presenceLeave() } catch {}
        try { ch.close() } catch {}
      }
    }
  }, [roomId, sub])

  const publish = useMemo(() => {
    return (event, data) => channelRef.current?.publish(event, { ...data, _from: getClientId() })
  }, [])

  const subscribe = useMemo(() => {
    return (event, handler) => channelRef.current?.subscribe(event, handler)
  }, [])

  const shareUrl = useMemo(() => {
    try {
      const url = new URL(window.location.href)
      url.searchParams.set('room', roomId)
      return url.toString()
    } catch {
      return ''
    }
  }, [roomId])

  return { roomId, publish, subscribe, users, shareUrl }
}
