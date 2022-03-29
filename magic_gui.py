# Entrypoint for pyinstaller

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from magic_afm.gui import main
    main()
else:
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

