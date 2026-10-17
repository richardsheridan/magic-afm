"""The GUI's colourbar tick labels carry an SI prefix only for units that take one."""

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

import re
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest
import trio

SHAPE = (64, 64)
LINEAR = np.linspace(0.0, 1.0, np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
PLAIN_NUMBER = re.compile(r"[-+]?\d+(\.\d*)?([eE][-+]?\d+)?")
# Units that take no SI prefix: each tick reads "<number> <unit>"
UNPREFIXED = {
    "ln(m)": LINEAR,
    "samples": 3000 * LINEAR,
    "°": LINEAR,
    "mN/m": LINEAR,
}
# SI unit symbols, with values that call for a prefix
PREFIXED = {
    "m": 3e-9 * LINEAR,
    "N": 3e-3 * LINEAR,
    "Pa": 3e9 * LINEAR,
    "V": 3e-3 * LINEAR,
    "A": 3e-3 * LINEAR,
    "s": 3e-3 * LINEAR,
    "Hz": 3e3 * LINEAR,
    "rad": 3e-3 * LINEAR,
}


def colourbar_ticks(unit, image, log_scale=False):
    """Show an image with the GUI's own drawing, headless; return its major ticks."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from magic_afm.gui import _impl as gui

    fig = Figure()
    FigureCanvasAgg(fig)
    display = MagicMock()
    display.fig = fig
    display.img_ax = fig.add_subplot()
    opened = gui.DemoForceVolumeFile()
    opened.add_image("Img", unit, image)
    controller = gui.ForceVolumeController(display, opened)

    async def draw():
        draw_send, draw_recv = trio.open_memory_channel(1)
        display.canvas.draw_send = draw_send
        async with trio.open_nursery() as nursery:
            image_send = await nursery.start(controller.change_image_task)
            image_send.send_nowait(("Img", "viridis", log_scale))
            with trio.fail_after(10):
                draw_fn = await draw_recv.receive()
            nursery.cancel_scope.cancel()
        draw_fn()

    trio.run(draw)
    axis = controller.axesimage.colorbar.ax.yaxis
    labels = [label.get_text() for label in axis.get_majorticklabels()]
    locs = list(axis.get_majorticklocs())
    assert len(labels) == len(locs) > 1
    return locs, labels


def assert_plain_number(text, loc):
    """The text is a number with no prefix that reads back as the tick's value."""
    number = text.replace("−", "-")  # matplotlib's U+2212 minus sign
    assert PLAIN_NUMBER.fullmatch(number), text
    last_place = Decimal(number).as_tuple().exponent
    assert isinstance(last_place, int)
    np.testing.assert_allclose(
        float(number), loc, rtol=0, atol=0.5 * 10.0**last_place + 1e-12 * abs(loc)
    )


@pytest.mark.parametrize("unit", [None, ""], ids=["None", "empty"])
def test_colourbar_of_a_unitless_image_is_labelled_with_plain_numbers(unit):
    locs, labels = colourbar_ticks(unit, LINEAR)
    for loc, label in zip(locs, labels):
        assert "None" not in label
        assert_plain_number(label, loc)


@pytest.mark.parametrize("unit", UNPREFIXED)
def test_colourbar_labels_a_unit_without_prefixes_as_plain_number_and_unit(unit):
    locs, labels = colourbar_ticks(unit, UNPREFIXED[unit])
    for loc, label in zip(locs, labels):
        assert label.endswith(" " + unit), label
        assert_plain_number(label.removesuffix(" " + unit), loc)


@pytest.mark.parametrize("unit", PREFIXED)
def test_colourbar_labels_an_si_unit_with_an_engineering_prefix(unit):
    """SI unit symbols keep the one-decimal engineering labels, e.g. "3.0 nm"."""
    from matplotlib.ticker import EngFormatter

    locs, labels = colourbar_ticks(unit, PREFIXED[unit])
    formatter = EngFormatter(unit, places=1)
    assert labels == [formatter(loc) for loc in locs]


@pytest.mark.parametrize(
    ("unit", "image"),
    [(None, 10.0 ** (-3 * LINEAR)), ("samples", 10.0 ** (4 * LINEAR))],
    ids=["None", "samples"],
)
def test_log_scaled_colourbar_labels_major_ticks_without_prefixes(unit, image):
    locs, labels = colourbar_ticks(unit, image, log_scale=True)
    suffix = "" if unit is None else " " + unit
    for loc, label in zip(locs, labels):
        assert "None" not in label
        assert label.endswith(suffix), label
        assert_plain_number(label.removesuffix(suffix), loc)
