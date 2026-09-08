"""Magic AFM GUI package

Only the standard library is imported here, so crash reporting still works
when importing the heavy GUI dependencies is what failed.
"""

# Copyright (C) Richard J. Sheridan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import sys


def report_crash():
    """Save the exception to a dump file, then try to show a dialog.

    Without a console nobody sees the traceback, and a macOS app launched
    from Finder runs with CWD "/", which is read-only.
    So, dump to ~/Library/Logs there.
    """
    import datetime
    import pathlib
    import traceback

    folder = pathlib.Path()
    if sys.platform == "darwin":
        folder = pathlib.Path.home() / "Library" / "Logs" / "MagicAFM"
        folder.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().isoformat().replace(":", ";")
    path = (folder / f"traceback-{date}.dump").absolute()
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