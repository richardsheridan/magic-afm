=========
Releasing
=========

1. Create the GitHub release and its tag
----------------------------------------

On GitHub, go to **Releases > Draft a new release**, type the new tag ``<version>``
(e.g., ``1.0.0b3``) under **Choose a tag**, write the release notes, and
publish. GitHub then creates the version tag for use in the remainder of the flow.

2. Build the executables
------------------------

Go to **Actions > Build > Run workflow**, pick the tag under **Use workflow
from**, and wait for both jobs (``windows-latest`` and ``macos-latest``) to finish.

3. Attach the executables to the release
----------------------------------------

From the workflow run's summary page, download and clean up both artifacts:

- ``magic_afm-windows-latest-build.zip``: rename it to
  ``magic_afm-windows-latest.zip``.
- ``magic_afm-macos-latest-build.zip``: extract it and keep the
  ``magic_afm_mac.zip`` inside only.

Edit the release and add both files as assets.

4. Upload to PyPI
-----------------

Build from a clean checkout of the tag and upload. Take care to choose the
correct ``<version>``!

.. code-block:: bash

    git fetch --tags
    git checkout <version>
    rm -rf dist/pypi
    pixi run -e build python -m build --outdir dist/pypi
    pixi run -e build twine check dist/pypi/*
    pixi run -e build twine upload dist/pypi/*
    git checkout master
