"""MagicAFM CLI

This is a command line interface to batch fit every force curve in a set of
files with the Magic AFM model and export the results. Run
`python -m magic_afm.cli --help` to see the options.

The CLI itself lives in _impl. Unlike the GUI, it fits in a plain
concurrent.futures process pool without trio. It reads back the options
JSON, so keep OPTIONS_JSON_SCHEMA in sync between the GUI and CLI when
adding fit or preprocessing parameters.
"""
