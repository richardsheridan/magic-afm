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

    proc = psutil.Process()
    try:
        try:
            NICE = psutil.BELOW_NORMAL_PRIORITY_CLASS
        except AttributeError:
            # POSIX forbids an unprivileged process from lowering its nice value,
            # so a worker started above 3 (under nice, a batch scheduler, some
            # agent shells) must keep its own value rather than ask for 3
            NICE = max(3, proc.nice())
        proc.nice(NICE)
    except psutil.AccessDenied:
        pass  # lowering priority is best effort, e.g. in a Windows sandbox


def cli_init(jit):
    nice_workers()
    if jit:
        import warnings

        from .calculation import _warmup_jit_worker

        # hacked NumbaExperimentalWarning to be just Warning to avoid slowdown bug
        warnings.simplefilter("ignore", Warning, lineno=621)
        _warmup_jit_worker()
    from numpy import seterr

    seterr(all="ignore")
