def nice_workers():
    import os
    import psutil
    import sys

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
