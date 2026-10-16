"""Smoke test of the real Tk GUI, run in a subprocess by tests/gui_smoke_driver.py."""

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

# This opens windows, so it runs only with MAFM_GUI_TESTS=1 (CI sets it, under
# xvfb-run on Linux). The app runs in a child process with real wall-clock
# time and real trio-parallel workers; the child bounds its own driving with
# trio.fail_after, the timeout here bounds the whole run including exit.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = Path(__file__).resolve().parent / "gui_smoke_driver.py"
CHILD_TIMEOUT = 300

pytestmark = pytest.mark.skipif(
    os.environ.get("MAFM_GUI_TESTS") != "1",
    reason="GUI test opens windows; set MAFM_GUI_TESTS=1 to run it",
)


def _text(captured):
    # TimeoutExpired carries bytes (or None) even when run() was given text=True
    if isinstance(captured, bytes):
        return captured.decode("utf-8", "replace")
    return captured or ""


def test_gui_demo_fit_then_quit_exits_0_with_finite_property_maps(tmp_path):
    """File > Demo, calculate property maps, File > Quit: exit 0, every pixel finite."""
    env = dict(os.environ)
    env["MAFM_MAX_CPUS"] = "2"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(DRIVER)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=tmp_path,
            timeout=CHILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(
            f"GUI did not exit within {CHILD_TIMEOUT} s\n"
            f"--- stdout ---\n{_text(e.stdout)}\n--- stderr ---\n{_text(e.stderr)}"
        )
    output = f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.returncode == 0, output

    finite_fractions = json.loads(proc.stdout.strip().splitlines()[-1])
    assert finite_fractions, output
    assert all(fraction == 1.0 for fraction in finite_fractions.values()), output
