if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # frozen workers run HERE then raise SystemExit
    try:
        from magic_afm.gui._impl import main
    except Exception:  # main() reports its own crashes
        from magic_afm.gui import report_crash

        report_crash()
        raise

    main()
