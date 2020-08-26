# Entrypoint for pyinstaller

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from magic_afm.gui import main
    main()
