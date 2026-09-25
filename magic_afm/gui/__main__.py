if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # frozen workers run HERE then raise SystemExit
    import os
    import sys

    from magic_afm.gui import report_crash, user_cache_dir

    if getattr(sys, "frozen", False):
        # PyInstaller points matplotlib at a new temp dir on every launch
        os.environ["MPLCONFIGDIR"] = str(user_cache_dir() / "matplotlib")
    try:
        from magic_afm.gui._impl import main
    except Exception:  # main() reports its own crashes
        report_crash()
        raise

    main()
