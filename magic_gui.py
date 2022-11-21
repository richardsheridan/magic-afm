# Entrypoint for pyinstaller

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # frozen workers run HERE then raise SystemExit
    del os.environ["OPENBLAS_NUM_THREADS"]
    from magic_afm.gui import main
    main()

