"""Worker initializers survive a priority drop the OS refuses."""

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

import os
import sys

import numpy as np
import psutil
import pytest

from magic_afm import _util

NUMPY_WARNING = "ignore:Somehow numpy was imported:UserWarning"


@pytest.fixture
def denied_nice(monkeypatch):
    """Make every renice raise AccessDenied; return the list of attempted calls."""
    calls = []

    def fake_nice(self, value=None):
        calls.append(value)
        raise psutil.AccessDenied(os.getpid())

    monkeypatch.setattr(psutil.Process, "nice", fake_nice)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    return calls


@pytest.mark.filterwarnings(NUMPY_WARNING)
def test_nice_workers_returns_when_renice_is_denied(denied_nice):
    """nice_workers returns normally when psutil denies the priority change."""
    _util.nice_workers()
    assert denied_nice


def test_nice_workers_sets_openblas_limiter_when_renice_is_denied(
    monkeypatch, denied_nice
):
    """With numpy not yet imported, a denied renice still leaves the limiter set."""
    monkeypatch.delitem(sys.modules, "numpy")
    _util.nice_workers()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert denied_nice


@pytest.mark.filterwarnings(NUMPY_WARNING)
def test_cli_init_returns_when_renice_is_denied(denied_nice):
    """The CLI worker initializer returns normally when the renice is denied."""
    with np.errstate():
        _util.cli_init(jit=False)
    assert denied_nice


@pytest.mark.filterwarnings(NUMPY_WARNING)
def test_nice_workers_never_raises_priority_of_an_already_niced_process(monkeypatch):
    """Started at nice 10 on POSIX, nice_workers asks for nothing below 10."""
    set_calls = []

    def fake_nice(self, value=None):
        if value is None:
            return 10
        set_calls.append(value)
        if value < 10:
            raise psutil.AccessDenied(os.getpid())

    monkeypatch.delattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", raising=False)
    monkeypatch.setattr(psutil.Process, "nice", fake_nice)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    _util.nice_workers()
    assert all(value >= 10 for value in set_calls)
