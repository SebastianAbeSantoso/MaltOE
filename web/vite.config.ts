import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Everything is bundled: the app must work with no network (Req 8.3), so no CDN links, fonts, or scripts.
export default defineConfig({
  plugins: [react()],
  server: {
    // `npm run dev` talks to molt-core on the loopback interface.
    proxy: { '/api': 'http://127.0.0.1:8000', '/ws': { target: 'ws://127.0.0.1:8000', ws: true } },
  },
  build: { outDir: 'dist' },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
  },
})
