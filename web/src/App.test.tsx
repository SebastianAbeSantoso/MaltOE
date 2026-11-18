import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

afterEach(() => vi.unstubAllGlobals())

describe('App', () => {
  it('shows the server version when the health check succeeds', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'ok', version: '0.1.0' }) }))
    render(<App />)
    expect(await screen.findByText(/Local server is running \(version 0\.1\.0\)/)).toBeInTheDocument()
  })

  it('says so when the server is not reachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network error')))
    render(<App />)
    expect(await screen.findByText(/not reachable/)).toBeInTheDocument()
  })

  it('treats an error status as unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({}) }))
    render(<App />)
    expect(await screen.findByText(/not reachable/)).toBeInTheDocument()
  })

  it('always shows the product name and tagline', () => {
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(new Promise(() => {})))
    render(<App />)
    expect(screen.getByRole('heading', { name: 'MaltOE' })).toBeInTheDocument()
    expect(screen.getByText('Room to grow.')).toBeInTheDocument()
  })
})
