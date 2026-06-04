from __future__ import annotations

import subprocess


def cuda_summary() -> dict:
    try:
        import torch

        return {
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        return {"cuda_available": False, "error": repr(exc)}


def require_cuda() -> None:
    summary = cuda_summary()
    if not summary.get("cuda_available"):
        raise RuntimeError(
            "CUDA is not available to this Python process. Run outside the sandbox "
            "or set CUDA_VISIBLE_DEVICES before launching Track 1."
        )


def nvidia_smi_text() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc!r}"
