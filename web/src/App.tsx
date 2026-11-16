import { useEffect, useState } from 'react'

type Health = { status: string; version: string }
type ServerState = { kind: 'checking' } | { kind: 'ok'; version: string } | { kind: 'unreachable' }

// The chat panel and the live dashboard arrive in Task 11. This shell only proves that the
// toolchain works and that the page can reach molt-core.
export default function App() {
  const [server, setServer] = useState<ServerState>({ kind: 'checking' })

  useEffect(() => {
    let cancelled = false
    fetch('/api/health')
      .then((r) => (r.ok ? (r.json() as Promise<Health>) : Promise.reject(new Error(String(r.status)))))
      .then((h) => !cancelled && setServer({ kind: 'ok', version: h.version }))
      .catch(() => !cancelled && setServer({ kind: 'unreachable' }))
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main>
      <h1>Molt</h1>
      <p>One model, any GPU — it molts to fit.</p>
      <p role="status">
        {server.kind === 'checking' && 'Checking the local server…'}
        {server.kind === 'ok' && `Local server is running (version ${server.version}).`}
        {server.kind === 'unreachable' && 'The local server is not reachable.'}
      </p>
    </main>
  )
}
