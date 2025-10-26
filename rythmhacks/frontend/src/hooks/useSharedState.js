import { useEffect, useRef } from 'react'
import { getClientId } from '../realtime/ablyClient'

// Synchronize a piece of state across users in a room
// publish/subscribe come from useRoom()
export default function useSharedState({ key, value, setValue, publish, subscribe }) {
  const lastFromRef = useRef(null)

  // Publish on local change
  useEffect(() => {
    if (!publish) return
    publish(`state:${key}`, { key, value, ts: Date.now() })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  // Receive remote updates
  useEffect(() => {
    if (!subscribe) return
    const handler = (data) => {
      if (!data) return
      const { value: v, _from } = data
      if (_from && _from === getClientId()) return // ignore echoes
      try {
        setValue(v)
      } catch {}
    }
    subscribe(`state:${key}`, handler)
    // No cleanup needed with our channel wrapper
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subscribe, key, setValue])
}

