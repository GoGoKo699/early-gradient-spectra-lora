"""RMT-LoRA simulation package.

This package contains pure numerical experiments for testing whether LoRA-like
low-rank adaptation is better explained by detectable spectral rank than by
nominal rank.
"""

# Small/medium SVD-heavy simulations often run faster and more reproducibly with
# one BLAS thread. Users can override these environment variables externally.
import os as _os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
):
    _os.environ.setdefault(_name, "1")

try:  # Dynamic control for sessions where NumPy/OpenBLAS was already loaded.
    from threadpoolctl import threadpool_limits as _threadpool_limits

    _THREADPOOL_LIMITS = _threadpool_limits(limits=1, user_api="blas")
except Exception:  # pragma: no cover - optional runtime optimization only.
    _THREADPOOL_LIMITS = None

__version__ = "0.1.0"
