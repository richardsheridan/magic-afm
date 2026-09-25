"""MagicAFM GUI

This is a trio guest-mode async tkinter graphical interface for AFM users to
calculate indentation ratios and modulus sensitivities for their force curve
data in an intuitive and responsive package. Launch it with
`python -m magic_afm.gui`.

The GUI itself lives in _impl. This module holds only the app metadata and the
crash reporting helpers, so it must import nothing but the standard library:
reporting a crash has to work even when importing the GUI fails.
This package should not be imported in a worker process.
"""

__author__ = "Richard J. Sheridan"
__app_name__ = __doc__.split("\n", 1)[0]

# noinspection PyUnreachableCode
if __debug__:
    from multiprocessing import parent_process

    assert parent_process() is None, "importing gui code in a worker"

# FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("magic-afm")
except PackageNotFoundError:
    try:
        from magic_afm._version import __version__
    except ImportError:
        __version__ = "(unknown version)"

__short_license__ = f"""{__app_name__} {__version__}
Copyright (C) {__author__}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# stdlib imports ONLY
import os
import pathlib
import sys

APP_NAME = "MagicAFM"


def user_cache_dir():
    """Per-user cache folder, like platformdirs.user_cache_dir(APP_NAME)"""
    return _user_dir("Caches", "Cache", "XDG_CACHE_HOME", ".cache")


def user_log_dir():
    """Per-user log folder, like platformdirs.user_log_dir(APP_NAME)"""
    return _user_dir("Logs", "Logs", "XDG_STATE_HOME", ".local/state")


def _user_dir(macos, windows, xdg_var, xdg_default):
    # a stdlib stand-in for platformdirs
    home = pathlib.Path.home()
    if sys.platform == "darwin":
        return home / "Library" / macos / APP_NAME
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local"
        return pathlib.Path(base) / APP_NAME / windows
    base = os.environ.get(xdg_var) or home / xdg_default
    return pathlib.Path(base) / APP_NAME


def report_crash():
    """Save the exception to a dump file, then try to show a dialog.

    Without a console nobody sees the traceback, and a macOS app launched
    from Finder runs with CWD "/", which is read-only.
    So, dump to the per-user log folder.
    """
    import datetime
    import traceback

    folder = user_log_dir()
    folder.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().isoformat().replace(":", ";")
    path = folder / f"traceback-{date}.dump"
    with path.open("w", encoding="utf8") as file:
        traceback.print_exc(file=file)

    try:
        import tkinter as tk
        import tkinter.messagebox

        root = tk.Tk()
        root.withdraw()
        tkinter.messagebox.showerror(
            "Magic AFM",
            f"Magic AFM hit an unexpected error and must close.\n\n"
            f"Details were saved to:\n{path}",
            parent=root,
        )
        root.destroy()
    except Exception:
        pass