"""Real-file value goldens are chosen per platform; synthetic goldens are shared."""

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

# Real-file fits are reproducible on one platform but differ between them
# (RELEASE_PLAN.md, M0 findings, 2026-10-16), so their goldens live in
# tests/goldens/<pixi platform>/ under the shared files' names. These tests
# fake the platform and need no real data.

import json
import platform
import sys
from pathlib import Path

import conftest
import pytest

SHARED_DIR = Path(__file__).resolve().parent / "goldens"


def write_set(root, plat, name, content):
    path = root / plat / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


@pytest.mark.parametrize(
    ("sys_platform", "machine", "expected"),
    [
        ("darwin", "arm64", "osx-arm64"),
        ("darwin", "x86_64", "osx-64"),
        ("linux", "x86_64", "linux-64"),
        ("linux", "aarch64", "linux-aarch64"),
        ("win32", "AMD64", "win-64"),
        ("win32", "ARM64", "win-arm64"),
    ],
)
def test_golden_platform_is_the_pixi_platform_name(
    monkeypatch, sys_platform, machine, expected
):
    monkeypatch.setattr(sys, "platform", sys_platform)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert conftest.golden_platform() == expected


def test_platform_goldens_are_the_set_for_the_running_platform(monkeypatch, tmp_path):
    for plat in ("osx-arm64", "linux-64", "win-64"):
        write_set(tmp_path, plat, "t2_goldens.json", {"platform": plat})
    monkeypatch.setattr(conftest, "GOLDENS_DIR", tmp_path)
    monkeypatch.setattr(conftest, "golden_platform", lambda: "linux-64")
    got = conftest.load_platform_goldens("t2_goldens.json")
    assert got == {"platform": "linux-64"}


def test_platform_goldens_skip_naming_a_platform_without_a_set(monkeypatch, tmp_path):
    write_set(tmp_path, "osx-arm64", "t2_goldens.json", {})
    monkeypatch.setattr(conftest, "GOLDENS_DIR", tmp_path)
    monkeypatch.setattr(conftest, "golden_platform", lambda: "linux-aarch64")
    monkeypatch.delenv("MAFM_REQUIRE_REAL_DATA", raising=False)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        conftest.load_platform_goldens("t2_goldens.json")
    assert "linux-aarch64" in excinfo.value.msg
    assert "make_t" in excinfo.value.msg  # the generator scripts


def test_platform_goldens_fail_without_a_set_when_real_data_is_required(
    monkeypatch, tmp_path
):
    write_set(tmp_path, "osx-arm64", "t2_goldens.json", {})
    monkeypatch.setattr(conftest, "GOLDENS_DIR", tmp_path)
    monkeypatch.setattr(conftest, "golden_platform", lambda: "linux-aarch64")
    monkeypatch.setenv("MAFM_REQUIRE_REAL_DATA", "1")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        conftest.load_platform_goldens("t2_goldens.json")
    assert "linux-aarch64" in excinfo.value.msg


def test_shared_goldens_hold_no_real_file_values():
    """Synthetic T2 values and T3's options stay shared; real-file values move."""
    t2 = json.loads((SHARED_DIR / "t2_goldens.json").read_text())
    t3 = json.loads((SHARED_DIR / "t3_goldens.json").read_text())
    assert "synthetic" in t2 and "nid" not in t2
    assert "options" in t3 and not {"summaries", "full"} & t3.keys()


def test_osx_arm64_golden_set_holds_the_real_file_values():
    plat_dir = SHARED_DIR / "osx-arm64"
    t2 = json.loads((plat_dir / "t2_goldens.json").read_text())
    t3 = json.loads((plat_dir / "t3_goldens.json").read_text())
    assert set(t2["nid"]) >= {"EXTEND", "RETRACT", "BOTH"}
    assert {"summaries", "full"} <= t3.keys()
