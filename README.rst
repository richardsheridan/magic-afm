===========================================================
Magic AFM -- Code to analyze and understand Magic Ratio AFM
===========================================================
Companion software to the paper `Vanishing Cantilever Calibration Error with Magic
Ratio Atomic Force Microscopy <https://onlinelibrary.wiley.com/doi/abs/10.1002/adts.202000090>`_

----------
Motivation
----------
This software exists so that people can easily calculate indentation ratios and
modulus sensitivities for their forcecurve data.

------------
Installation
------------
Requires cpython 3.8 or higher and several custom patched packages

::

    git clone https://github.com/richardsheridan/magic-afm.git
    pip install -r requirements.txt

eventually setup.py?

Pyinstaller build available after openblas bugfix

If dealing with ARDF files, you must acquire ARDFtoHDF5.exe from Oxford
Instruments/Asylum Research and put it in the root of the pyinstaller
or git folder.


-----
Usage
-----
Most common usage: run magic_gui.exe on windows.

On other platforms, run python magic_gui.py. (Should work, but untested)

All functionality available when the source is imported

---
API
---
magic_afm can be imported and used, especially the submodules data_readers and
calculation

This section will list the function names, arguments, results, exceptions and
side effects. Possibly generated from docstrings?