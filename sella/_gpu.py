"""Centralized GPU helpers for Sella.

Provides numpy-in / numpy-out wrappers for eigh and QR that route through
torch+CUDA when a usable GPU is present, with a single shared `_has_torch`
state and a size threshold that gates upload overhead.

GPU usage is opt-out via SELLA_DISABLE_GPU=1 and the size threshold is
tunable via SELLA_GPU_MIN_DIM (default 200). On a torch CUDA OOM the call
falls back to CPU and records the failure so subsequent calls skip the GPU
attempt for that size; this keeps smaller GPUs from thrashing.
"""

import os
import numpy as np
from scipy.linalg import eigh as _cpu_eigh
from scipy.linalg import qr as _cpu_qr

try:
    import torch
    _has_torch = torch.cuda.is_available()
except Exception:
    torch = None
    _has_torch = False

if os.environ.get("SELLA_DISABLE_GPU", "").lower() in ("1", "true", "yes"):
    _has_torch = False

try:
    _GPU_MIN_DIM = int(os.environ.get("SELLA_GPU_MIN_DIM", "200"))
except ValueError:
    _GPU_MIN_DIM = 200

# After a CUDA OOM at dimension N, refuse subsequent GPU offload for shapes
# >= N. Keeps a single failure from cascading and lets the CPU path take over
# cleanly on small GPUs.
_oom_floor = None


def _gpu_ok(n):
    return _has_torch and n >= _GPU_MIN_DIM and (
        _oom_floor is None or n < _oom_floor
    )


def _record_oom(n):
    global _oom_floor
    if _oom_floor is None or n < _oom_floor:
        _oom_floor = n
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def to_gpu(A):
    """Upload a numpy array to GPU as a contiguous float64 tensor.

    Returns the torch tensor, or None if no GPU is available / upload fails.
    Caller is responsible for size-gating with `_gpu_ok` if desired.
    """
    if not _has_torch:
        return None
    try:
        return torch.from_numpy(np.ascontiguousarray(A)).cuda()
    except (RuntimeError, MemoryError):
        _record_oom(A.shape[0] if A.ndim >= 1 else 0)
        return None


def gpu_eigh(A, A_gpu=None):
    """Eigendecomposition. GPU when beneficial, CPU otherwise.

    If A_gpu (a torch tensor on CUDA) is supplied, the upload is skipped.
    """
    n = A.shape[0]
    if n == 0:
        # scipy.linalg.eigh rejects 0x0 input (LAPACK dsyevr requires il<=n),
        # whereas np.linalg.eigh returns empties. Match numpy so callers with
        # a fully-constrained (empty) subspace don't crash.
        return (np.empty(0, dtype=np.float64),
                np.empty((0, 0), dtype=np.float64))
    if A_gpu is not None or _gpu_ok(n):
        try:
            At = A_gpu if A_gpu is not None else to_gpu(A)
            if At is not None:
                evals_t, evecs_t = torch.linalg.eigh(At)
                return evals_t.cpu().numpy(), evecs_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return _cpu_eigh(A)


def gpu_eigh_t(A_gpu):
    """Eigendecomposition that returns torch tensors (no .cpu() download).

    Caller has already uploaded; returns (evals_t, evecs_t) on GPU, or
    (None, None) on OOM so caller can fall back.
    """
    try:
        return torch.linalg.eigh(A_gpu)
    except (RuntimeError, MemoryError):
        _record_oom(A_gpu.shape[0])
        return None, None


def gpu_qr(A):
    """Economy QR. GPU when beneficial, CPU otherwise."""
    n = A.shape[0]
    if _gpu_ok(n):
        try:
            At = to_gpu(A)
            if At is not None:
                Q_t, R_t = torch.linalg.qr(At, mode='reduced')
                return Q_t.cpu().numpy(), R_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    if n == 0:
        return np.linalg.qr(A, mode='reduced')
    return _cpu_qr(A, mode='economic', pivoting=False, check_finite=False)


def gpu_svd(M, full_matrices=True):
    """Economy/full SVD on GPU when beneficial, else None.

    Returns (U, S, Vh) as numpy arrays (same convention as numpy.linalg.svd),
    or ``None`` when the GPU path is unavailable/too small/failed so the caller
    can run its own CPU path (e.g. the gesdd->gesvd robust fallback). Note
    cuSOLVER can itself raise on non-convergence (torch.linalg.LinAlgError, a
    RuntimeError subclass) -- that is caught here and reported as None so the
    caller's robust CPU driver takes over.
    """
    n = M.shape[0]
    if _gpu_ok(n):
        try:
            Mt = to_gpu(M)
            if Mt is not None:
                U_t, S_t, Vh_t = torch.linalg.svd(Mt, full_matrices=full_matrices)
                return U_t.cpu().numpy(), S_t.cpu().numpy(), Vh_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return None


def gpu_pinv(B, rcond=1e-15):
    """Moore-Penrose pseudoinverse on GPU when beneficial, else None.

    Computed via GPU SVD with the same relative cutoff (rcond * largest sv) as
    numpy.linalg.pinv, so results match the CPU path to machine precision in
    fp64. Returns a numpy array, or ``None`` when the GPU path is
    unavailable/too small/failed (caller falls back to CPU).
    """
    n = B.shape[0]
    if _gpu_ok(n):
        try:
            Bt = to_gpu(B)
            if Bt is not None:
                U_t, S_t, Vh_t = torch.linalg.svd(Bt, full_matrices=False)
                cutoff = rcond * S_t[0] if S_t.numel() else S_t.new_zeros(())
                Sinv = torch.where(S_t > cutoff, 1.0 / S_t,
                                   torch.zeros_like(S_t))
                P_t = (Vh_t.transpose(-2, -1) * Sinv) @ U_t.transpose(-2, -1)
                return P_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return None


def gpu_project(H, U, H_gpu=None):
    """Compute U.T @ H @ U on GPU when beneficial.

    Returns a numpy array. If H_gpu is supplied (cached), the H upload is
    skipped — this is the main motivation for the helper, since the same H
    is read by both the BFGS eigh and the projection in the same step.
    """
    n = H.shape[0]
    if H_gpu is not None or _gpu_ok(n):
        try:
            Ht = H_gpu if H_gpu is not None else to_gpu(H)
            Ut = to_gpu(U)
            if Ht is not None and Ut is not None:
                # Project on device, download only the (k, k) result.
                R_t = Ut.T @ Ht @ Ut
                return R_t.cpu().numpy()
        except (RuntimeError, MemoryError):
            _record_oom(n)
    return U.T @ H @ U
