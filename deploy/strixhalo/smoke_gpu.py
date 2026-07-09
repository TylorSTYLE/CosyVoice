#!/usr/bin/env python3
"""Phase 1 gfx1151 smoke test — run INSIDE the container ON the Strix Halo server.

Gate: no SIGSEGV (exit 139), torch.cuda.is_available() True, GPU matmul correct.
Also establishes the autoregressive-decode bottleneck baseline that Phase 2 vLLM
must beat: the "sync overhead" ratio proxies the gfx1151 hipMemcpyWithStream cost
(PyTorch issue #171687) — a per-decode-step device->host copy.
"""
import time
import torch


def main() -> None:
    print("=== gfx1151 torch smoke ===")
    print("torch:", torch.__version__, "| hip:", getattr(torch.version, "hip", None))
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: torch.cuda.is_available() is False — ROCm/torch not seeing gfx1151")
    dev = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0))

    # 1) correctness: GPU matmul vs CPU reference
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    c = a @ b
    torch.cuda.synchronize()
    err = (c.cpu() - (a.cpu() @ b.cpu())).abs().max().item()
    print(f"[matmul] max abs err vs CPU: {err:.3e}  ({'OK' if err < 1e-1 else 'FAIL'})")

    # 2) AR-decode proxy: many small GEMV steps, with vs without a per-step d2h sync.
    #    hidden/vocab roughly sized to a CosyVoice2/3 0.5B speech-token head.
    def bench(sync_each: bool, steps: int = 512, hidden: int = 896, vocab: int = 6561):
        W = torch.randn(vocab, hidden, device=dev)
        x = torch.randn(hidden, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            logits = W @ x          # (vocab,)
            nxt = torch.argmax(logits)
            if sync_each:
                _ = nxt.item()      # forces device->host copy every step (the pathology)
            x = x * 0.999 + 1e-3    # keep the loop data-dependent
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return steps / dt, dt / steps * 1e3

    # warmup
    bench(False, steps=64)
    tps_sync, ms_sync = bench(True)
    tps_nosync, ms_nosync = bench(False)
    print(f"[AR proxy] per-step d2h sync : {tps_sync:9.1f} tok/s  ({ms_sync:.3f} ms/tok)")
    print(f"[AR proxy] no per-step sync  : {tps_nosync:9.1f} tok/s  ({ms_nosync:.3f} ms/tok)")
    ratio = tps_nosync / tps_sync if tps_sync else float("nan")
    print(f"[AR proxy] sync overhead x   : {ratio:.2f}  "
          f"(>> 1 => hipMemcpyWithStream bottleneck present — Phase 2 vLLM must beat this)")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
