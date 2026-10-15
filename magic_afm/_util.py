import os
import sys

import psutil

from ._env import max_workers

MAX_WORKERS = max_workers()


def nice_workers():
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
        import warnings

        from .calculation import warmup_jit_worker

        # hacked NumbaExperimentalWarning to be just Warning to avoid slowdown bug
        warnings.simplefilter("ignore", Warning, lineno=621)
        warmup_jit_worker()
    from numpy import seterr

    seterr(all="ignore")
