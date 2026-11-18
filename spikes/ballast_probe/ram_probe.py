#!/usr/bin/env python3
"""Task 1.8 (gate G4), RAM half: can a process take a chosen amount of RAM so that the
SSD tier is exercised on a machine with plenty of memory (the 32 GB PC)?

Commits and touches N GiB, locks it with VirtualLock on Windows so the OS cannot page it out,
and reports how much RAM the OS says is still available (what MaltOE will size its RAM tier from).

Usage:
    uv run --with psutil python ram_probe.py 12          # hold 12 GiB until Enter is pressed
    uv run --with psutil python ram_probe.py 12 --seconds 60

On non-Windows systems it falls back to touching a bytearray, so the script can be checked
anywhere. Only the Windows numbers answer the gate.
"""
import argparse
import sys
import time

import psutil

GiB = 1024 ** 3


def avail() -> float:
    return psutil.virtual_memory().available / GiB


def hold_windows(n_bytes: int):
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.restype = wintypes.LPVOID
    k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.VirtualLock.argtypes = [wintypes.LPVOID, ctypes.c_size_t]
    k32.VirtualLock.restype = wintypes.BOOL
    k32.SetProcessWorkingSetSize.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
    k32.GetCurrentProcess.restype = wintypes.HANDLE

    MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE = 0x1000, 0x2000, 0x04
    # VirtualLock can only lock what fits in the working set, so raise the limit first.
    ok = k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), n_bytes + (256 << 20), n_bytes + (512 << 20))
    if not ok:
        print(f"SetProcessWorkingSetSize failed (error {ctypes.get_last_error()}); locking may fail")

    addr = k32.VirtualAlloc(None, n_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    if not addr:
        raise MemoryError(f"VirtualAlloc failed (error {ctypes.get_last_error()})")
    ctypes.memset(addr, 0xA5, n_bytes)  # touch every page
    locked = bool(k32.VirtualLock(addr, n_bytes))
    return addr, locked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gib", type=float)
    ap.add_argument("--seconds", type=int, default=0, help="hold this long; default: until Enter")
    args = ap.parse_args()
    n = int(args.gib * GiB)

    total = psutil.virtual_memory().total / GiB
    print(f"total RAM {total:.1f} GiB, available before {avail():.2f} GiB")
    held = None
    if sys.platform == "win32":
        held, locked = hold_windows(n)
        print(f"VirtualAlloc ok; VirtualLock {'ok' if locked else 'FAILED (pages may be paged out)'}")
    else:
        print("not Windows: touching a bytearray instead (this does not answer the gate)")
        held = bytearray(n)
        for i in range(0, n, 4096):
            held[i] = 1
    time.sleep(1.0)
    print(f"available after  {avail():.2f} GiB (held {args.gib:.2f} GiB)")
    if args.seconds > 0:
        time.sleep(args.seconds)
    else:
        input("holding; press Enter to release ")
    del held
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
