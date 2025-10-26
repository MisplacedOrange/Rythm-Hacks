import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    'import.meta.env.VITE_ABLY_KEY': JSON.stringify('R_H9Zw.dAVlcw:c7dbFN6FGPXNtGvbXN2OVmMSHmMko4TrnR6JD-57sLA')
  }
})
