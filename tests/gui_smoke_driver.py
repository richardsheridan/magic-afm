"""Drive the real Tk app: File > Demo, fit property maps, File > Quit."""

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

# Run by tests/test_gui_smoke.py in a subprocess; prints one JSON line mapping
# each Calc* image the fit wrote to its fraction of finite pixels. trio-parallel
# workers re-import this file as __mp_main__, so the top level stays stdlib-only
# and the GUI is imported only under the __main__ guard.

import json
import os
import sys

DRIVE_TIMEOUT = float(os.environ.get("MAFM_GUI_DRIVE_TIMEOUT", "240"))


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


async def poll(fn, interval=0.05):
    import trio

    while not (value := fn()):
        await trio.sleep(interval)
    return value


def step(text):
    print(f"gui_smoke_driver: {text}", file=sys.stderr, flush=True)


async def drive(root, impl, controllers):
    from tkinter import ttk

    import numpy as np
    import trio

    # the deadline covers main_task's exit too, so a Quit that does not end the
    # app fails here with a traceback rather than at the parent's timeout
    with trio.fail_after(DRIVE_TIMEOUT):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(impl.main_task, root)
            menubar = await poll(
                lambda: root["menu"] and root.nametowidget(root["menu"])
            )
            file_menu = root.nametowidget(menubar.entrycget("File", "menu"))
            step("File > Demo")
            file_menu.invoke("Demo")

            # calc_prop_map_response needs the image drawn by change_image_task
            ctl = await poll(
                lambda: (
                    controllers
                    and controllers[0].axesimage is not None
                    and controllers[0]
                )
            )
            (extend,) = (
                w
                for w in descendants(root)
                if isinstance(w, ttk.Radiobutton) and w.cget("text") == "Extend"
            )
            extend.invoke()
            button = ctl.display.calc_props_button
            assert str(button.cget("state")) == "normal", button.cget("state")
            step("Calculate Property Maps")
            button.invoke()

            cache = ctl.opened_fvol._image_cache
            await poll(lambda: any(name.startswith("Calc") for name in cache))
            result = {
                name: float(np.isfinite(image).mean())
                for name, image in cache.items()
                if name.startswith("Calc")
            }
            step("File > Quit")
            file_menu.invoke("Quit")
    step("main_task returned")
    return result


def main():
    import tkinter as tk
    from tkinter import ttk

    from magic_afm.gui import _impl

    # observe only: record each controller so the driver can find the Demo's
    controllers = []
    control_task = _impl.ForceVolumeController.control_task

    async def recording_control_task(self):
        controllers.append(self)
        await control_task(self)

    _impl.ForceVolumeController.control_task = recording_control_task

    # root setup as in _impl.main, without its crash dialog
    tk.NoDefaultRoot()
    root = tk.Tk()
    root.wm_resizable(False, False)
    root.wm_minsize(300, 20)
    root.wm_title("Magic AFM")
    if root.tk.call("tk", "windowingsystem") == "x11":
        ttk.Style(root).theme_use("clam")

    result = _impl.TkHost(root).run(drive, root, _impl, controllers)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    sys.exit(main())
