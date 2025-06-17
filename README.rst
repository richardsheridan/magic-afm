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

- QNM and FFV data from Nanoscope V3 or higher (Bruker/Veeco .spm/.pfc)

- Asylum research ARDF/HDF5

- Additional format contributions welcome!

Motivation
----------
This package is companion software to our recent publication, "Vanishing
Cantilever Calibration Error with Magic Ratio Atomic Force Microscopy" [MRAFM]_,
where we show that the ratio of cantilever deflection to tip indentation is an
important dimensionless number controlling the amplification or suppression of
sources of error, and that under common thermal calibration methods the
sensitivity of modulus estimates to the spring constant error can be eliminated
through choice of system parameters.

We hoped to create a force curve analysis package that helps AFM users to
calculate indentation ratios and modulus sensitivities for their force curve
data in an intuitive and responsive package.  By facilitating these sorts of
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
Requires CPython 3.11 or higher. (As of v0.9.0, we can read ARDF files natively,
so :code:`ARDFtoHDF5.exe` is no longer required. We can still read the converted
HDF5 files, if necessary.)

From source
^^^^^^^^^^^
Clone the git repository to your local machine, and
install into a fresh environment::

    git clone https://github.com/richardsheridan/magic-afm.git

    python -m venv magic_afm
    magic_afm\Scripts\activate.bat (or) magic_afm\bin\activate
    pip install -e .[gui,numba]
    (or)
    conda create -n magic_afm python=3.11
    conda activate magic_afm
    pip install -e .[gui,numba]

This is also a `Pixi <https://pixi.sh/>`_ project, so if you install Pixi, the
Python install is automatic and fast.


From PyPI
^^^^^^^^^
Eventually we should supply a pypi distribution.

From GitHub Releases
^^^^^^^^^^^^^^^^^^^^
We use PyInstaller to build executable releases for Windows on Github.
Simply download the ZIP archive, extract it to your hard drive.

Usage
-----
All functionality available when the source is imported from the :code:`magic_afm`,
see API_. :code:`magic fitting workflow.ipynb` doubles as an explainer and alternative
interface. It also functions as the test suite for the calculation code, such as it is.
The notebook needs additional dependencies you can get with the :code:`notebook` extra.

Running the GUI
^^^^^^^^^^^^^^^

All platforms, when running from source or PyPI::

     python -m magic_afm.gui

or if you are a Pixi user, you can use this while in the source tree::

     pixi run gui

When running the GitHub release executable for Windows::

     magic_gui.exe

To load data, select "Open..." from the File menu, or press Ctrl+O. A dialog
will open that will allow you to navigate and select any supported filetype.
This will open a data window and populate a number of options in the main window.
Many files and windows can be open simultaneously; the main window options will
display/affect the attributes of the *last selected* data window.

Viewing Images
^^^^^^^^^^^^^^

By default a height map is shown, when it is available inside the data file.
Other precalculated images can be displayed using the "Image" drop-down box. The
"Colormap" menu allows you to select from pre-defined colormaps. The
"logarithmic scale" checkbox can provide contrast when image
data varies over many orders of magnitude. Images can be flattened, offset, or
smoothed in the "Manipulations" window. The manipulated images are added to the
image menu with "Calc<Ext/Ret/Both><Manip>" where <Manip> is the selected operation
and <Ext/Ret/Both> indicates whether the parameter was estimated from the extend,
retract, or both data. The available image manipulations are:

Flatten
   A linear fit to each row of the image is subtracted from that row.

PlaneFit
   A 2D linear fit to the image is subtracted from that image.

Offset
   The minimum value present in the image is subtracted from that image.

Median3x1
   Each pixel is replaced by the median value of that pixel and its vertical
   neighbors. Good for removing scanline artifacts.

Median3x3
   Each pixel is replaced by the median value of that pixel and its eight
   neighbors. Good for removing extreme outlier pixels.

Gauss3x3
   Blurs the image with a small Gaussian kernel. Good for random (pixel-wise)
   noise reduction via spatial averaging.

FillNaNs
   If the fitter was unsuccessful at a certain point, it will write NaN values
   into those pixels so they appear bright red in the figure. This fills in
   those pixels selectively with the median of their non-NaN neighbors. Works
   best when bad fits are sparse!

All calculated images can be exported by clicking "Export calculated maps" to various
image and data formats. The current preprocessing and fit parameters will be written
as JSON in the same folder.

Viewing Force Curves
^^^^^^^^^^^^^^^^^^^^

Force curves are displayed by left-clicking the image in the data window.
Shift+click allows multiple curves to be plotted. Ctrl+drag plots
continuously as the mouse moves over the image. A cross
is displayed over the selected point/pixel and the plot is displayed in the
adjacent axes with the extend and retract curves in blue and orange,
respectively. The data are in absolute units, as recorded in the raw data,
without any offsets/shifts. This view provides a quick qualitative check on a
force curve.

The "Force curve display" can be toggled between spatial (d vs z) temporal (d vs t)
and natural (f vs δ) units. "Select data source" appears if we detect that the file
contains Trace and or Retrace data, so you can select which curves to display and fit.
The "Preprocessing" parameters are read from the data file metadata but can be
adjusted on the fly, updating the display immediately. "Deflection Sens." refers to
the calibration factor that multiplies the static vertical photodetector signal to
obtain the cantilever deflection in nm (sometimes called InvOLS.) "Spring Constant"
refers to the static cantilever spring constant measured at the position of the probe
tip. "Sync Distance" is only available when dealing with QNM data, and phase-corrects
the position of the peak force with respect to time and z position.

Fitting Data
^^^^^^^^^^^^

Fitting can be toggled between the default nothing (Skip), the approach curve (Extend) or
the retract curve (Retract), or simultaneously fit the extend and retract curves (Both).
The fit parameters are not read from the file and
only affect the display when either the extend or the retract portions of the
force curve are toggled to fit. There are a variety of parameters that can be optimized
against the data or fixed to a constant. The Schwarz [SCHWARZ]_ Model parameters are:

log10(M (Pa))
    The base-ten logarithm of the indentation modulus M=4/3*E/(1-ν*ν)

Tip Radius (nm)
    The nominal radius of the parabolic probe assumed in the indentation model. Tip
    radius and modulus are practically indistinguishable based on data. Therefore, one
    must be fixed, and traditionally that one is radius.

DMT-JKR (0-1)
    The transition parameter between the long-range and short-range adhesion
    force regimes. Formally, it is the ratio of the short-range work of adhesion to
    the total work of adhesion (τ1*τ1 in [SCHWARZ]_). In any ambiguous cases, this
    parameter is practically unidentifiable based on data so it is fixed by default.
    In typical usage simply put 0 for DMT or 1 for JKR behavior.

Lennard-Jones
    A log-transform of a normalized distance scaling parameter for describing long-range
    attractive forces.

There are also parameters available to subtract common simple artifacts from the deflection
signal:

Virtual Deflection
    A global linear slope of displacement with respect to piezo position.

Hydrodynamic Drag
    A global force offset proportional to the local piezo velocity.

Laser Intf. Period
    The period of the sinusoidal deflection offset in nanometers of piezo travel.
    Laser interference fitting is disabled if this period is 0.

Laser Intf. Ampl.
   The amplitude of the sinusoidal deflection offset in nanometers of deflection.
   Phase is also fit if the amplitude is not fixed to 0.

NOTE: the deflection artifacts are several times more computationally intensive to fit
compared to the Schwarz parameters, so they are disables (fixed to 0) by default.

The deflection and piezo displacement of all currently displayed force curves can
be exported by clicking "Export calculated force curves" to various text and binary formats.

If a fit has been performed, a table is displayed above the force curve indicating
the key inferred parameters:

M
    indentation modulus M=4/3*E/(1-ν*ν)

dM/dk x k/M
    relative sensitivity of M to the spring constant

F_adh
    force of adhesion

d
    cantilever deflection from the maximum to the snap-off minimum

δ
    probe indentation depth from the maximum to the snap-off minimum

d/δ
    indentation ratio

a_c
    contact radius at the maximum indentation depth

SSE
    sum of squared errors

Using this table you observe the best fit value and uncertainty for parameters
at any point in the map. Mainly, this helps diagnose issues and confirm robust
fits. If you select multiple points, the average of the values of those points
will be displayed. Note that the current calculation assumes you are using the GCI/GetReal/Qf1.3
calibration method, as is current best practice. If you are doing hard-contact +
thermal calibration, you must approximately double the relative sensitivity value.
For calibrations that do not involve the equipartition theorem, the sensitivity
value reported is not applicable.

Additionally, fitting plots a curve labeled "Model" for the best-fit estimate.
If viewing in d vs z mode, "Surface Z" indicates the apparent height of the
substrate after accounting for indentation effects. If viewing in f vs δ mode,
Max/Crit markers indicate the apparent point of the "Maximum" and "Critical"
(snap-off) force and indentation, respectively.

The "calculate properties" button rapidly fits all data in the file and
creates new images for each in the "Image" menu. All calculated property maps can
be exported like any other image by clicking "Export calculated maps". Again, The
current preprocessing and fit parameters will be written as JSON in the same folder.

.. TODO: establish if you are in the magic ratio regime

Future Plans:

- Test suite and CI

- Viscoelastic model

- CLI for batch fit

API
---
:code:`magic_afm` can be imported and used, especially the submodules :code:`data_readers` and
:code:`calculation`.

.. TODO: This section will list the function names, arguments, results, exceptions and
   side effects. Possibly generated from docstrings?

Contributing
------------
If you notice any bugs, need any help, or want to contribute any code,
GitHub issues and pull requests are very welcome!

If you are reporting a crash, please include the traceback dump that is written
in the source or PyInstaller folder.
