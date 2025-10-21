import psutil
import sys

MAX_WORKERS = psutil.cpu_count(logical=False) or 1


def nice_workers():
    import os

    if "numpy" in sys.modules:
        import warnings

        warnings.warn(
            "Somehow numpy was imported before setting the OpenBLAS limiter."
            "Expect some inefficiency due to core contention.",
            stacklevel=2,
        )
    else:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        NICE = psutil.BELOW_NORMAL_PRIORITY_CLASS
    except AttributeError:
        NICE = 3
    psutil.Process().nice(NICE)


def cli_init(jit):
    nice_workers()
    if jit:
        from .calculation import warmup_jit_worker

        warmup_jit_worker()
    from numpy import seterr

    seterr(all="ignore")
