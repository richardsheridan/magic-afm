if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # frozen workers run HERE then raise SystemExit
    from magic_afm.cli._impl import main

    main()
