"""MagicAFM CLI

This is a command line interface to batch fit every force curve in a set of
files with the Magic AFM model and export the results. Run
`python -m magic_afm.cli --help` to see the options.

The CLI itself lives in _impl. Unlike the GUI, it fits in a plain
concurrent.futures process pool without trio. It reads back the options
JSON that it and the GUI write, whose schema lives in magic_afm._options.
"""
