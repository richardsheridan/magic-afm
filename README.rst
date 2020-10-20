=====================================================
Magic AFM -- Force curve analysis for Magic Ratio AFM
=====================================================

Magic AFM is Free Software (see `LICENSE <LICENSE>`__) for the analysis of AFM
force curves (static force spectroscopy) on polymers and other soft materials.

Features:

- Robust force curve fitting. (Parabolic indenter with DMT/JKR/LJ adhesion via [SCHWARZ]_.)

- Responsive interface for direct visual inspection/exploration of curves and fits.

- Sensitivity analysis of modulus error, to help identify if imaging conditions
  are amplifying or suppressing systemic error.

Supported file formats:

- QNM, FV, and FFV data from Nanoscope V3 or higher (Bruker/Veeco .spm/.pfc)

- Asylum research ARDF/HDF5

  - ARDF usage requires ARDFtoHDF5.exe, see Installation_.

- Additional format contributions welcome!

Motivation
----------
This package is companion software to our recent publication, "Vanishing
Cantilever Calibration Error with Magic Ratio Atomic Force Microscopy" [MRAFM]_.
We hoped to create a force curve analysis package that helps AFM users to
calculate indentation ratios and modulus sensitivities for their force curve
data in an intuitive and responsive package. By facilitating these sorts of
calculations, we hope to improve the overall systematic error of reported
modulus maps in the greater AFM nanomechanics community.

.. [MRAFM] Sheridan, R. J., Collinson, D. W., & Brinson, L. C. (2020).
        "Vanishing Cantilever Calibration Error with Magic Ratio Atomic Force
        Microscopy." Advanced Theory and Simulations, 3(8), 2000090.
        `DOI:10.1002/adts.202000090 <https://doi.org/10.1002/adts.202000090>`__

.. [SCHWARZ] Schwarz, U. D. (2003). A generalized analytical model for the
        elastic deformation of an adhesive contact between a sphere and a flat
        surface. Journal of Colloid and Interface Science, 261(1), 99–106.
        `DOI:10.1016/S0021-9797(03)00049-3
        <https://doi.org/10.1016/S0021-9797(03)00049-3>`_

Installation
------------
Requires cpython 3.8 or higher and several custom patched packages. The easiest
installation method is to clone the git repository to your local machine, and
install from the requirements.txt into a fresh environment::

    git clone https://github.com/richardsheridan/magic-afm.git

    python -m venv magic_afm
    magic_afm\Scripts\activate.bat (or) magic_afm\bin\activate
    pip install -r requirements.txt
    (or)
    conda create -n magic_afm python=3.8
    conda activate magic_afm
    conda install --file requirements.txt

Eventually we should supply setup.py and a pypi package, once the dependencies
settle. A beta release Pyinstaller build will be available on Github after
an upstream bugfix in OpenBLAS.

If dealing with ARDF files, you must acquire ARDFtoHDF5.exe from Oxford
Instruments/Asylum Research and put it in the root of the source or Pyinstaller
folder. We then are able to automatically convert the ARDF files to HDF5
on demand. (NOTE: This will duplicate all each file's data on your hard drive!)

Usage
-----
All platforms, when running from source::

     python magic_gui.py

All functionality available when the source is imported from the :code:`magic_afm`,
see API_.

To load data, select "Open..." from the File menu, or press Ctrl+O. A dialog
will open that will allow you to navigate and select any supported filetype.
This will open a data window and populate a number of options in the main window.
Many files and windows can be open simultaneously; the main window options will
display/affect the attributes of the *last selected* data window.

By default a height map is shown, when it is available inside the data file.
Other precalculated images can be displayed using the "Image" drop-down box. The
"Colormap" menu allows you to select from a few pre-defined colormaps. Images
can be flattened or smoothed in the "Manipulations" window. The manipulated images
are added to the image menu with "Calc<Manip>" where <Manip> is the selected operation.

Force curves are displayed by left-clicking the image in the data window.
Shift+click allows multiple curves to be plotted. Ctrl+drag plots
continuously as the mouse moves over the image. A cross
is displayed over the selected point/pixel and the plot is displayed in the
adjacent axes with the extend and retract curves in blue and orange,
respectively. The data are in absolute units, as recorded in the raw data,
without any offsets/shifts. This view provides a quick qualitative check on a
force curve.

The "Force curve display" can be toggled between spatial (d vs z) and
natural (f vs δ) units. The "Preprocessing" parameters are read from the data file
metadata but can be adjusted on the fly, updating the display immediately. Fitting
can be toggled between the default nothing (Skip), the approach curve (Extend) or
the retract curve (Retract). The fit parameters are not read from the file and
only update the display when either the extend or the retract portions of the
force curve are toggled to fit. "Tip Radius (nm)" refers to the nominal radius
of the parabolic probe assumed in the indentation model. "DMT-JKR (0-1)" refers
to the transition parameter between the long-range and short-range adhesion
force regimes. Formally, it is the ratio of the short-range work of adhesion to
the total work of adhesion (τ1*τ1 in [SCHWARZ]_).

If a fit has been performed, a table is displayed above the force curve indicating
the key inferred parameters:

M
    Indentation Modulus

dM/dk x k/M
    relative sensitivity of M to the spring constant

F_adh
    Force of adhesion

d
    cantilever deflection

δ
    probe indentation depth

d/δ
    indentation ratio

Using this table you observe the best fit value and uncertainty for parameters
at any point in the map. Mainly, this helps diagnose issues and confirm robust
fits. The "calculate properties" button rapidly fits all data in the file and
creates new images for each in the "Image" menu.

.. establish if you are in the magic ratio regime

Future Plans:

- Viscolastic model

- Simultaneous extend/retract fit

API
---
:code:`magic_afm` can be imported and used, especially the submodules data_readers and
calculation. "magic fitting workflow.ipynb" doubles as an explainer and alternative
interface. It also functions as the test suite for the calculation code, such as it is.

.. This section will list the function names, arguments, results, exceptions and
   side effects. Possibly generated from docstrings?

Contributing
------------
If you notice any bugs, need any help, or want to contribute any code,
GitHub issues and pull requests are very welcome!