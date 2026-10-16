"""Tests for the shared real-data fixture that finds, unpacks and skips data files."""

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

import hashlib
import lzma
import os
import time
from pathlib import Path

import conftest
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RELPATH = "test_cases/SAMPLE/ffm/tiny0001.ARDF"


@pytest.fixture(autouse=True)
def private_cache(request, monkeypatch, tmp_path_factory):
    """Unpack synthetic payloads into a per-test cache, never the shared one.

    Tests marked slow read the real corpus and keep the shared cache.
    """
    if request.node.get_closest_marker("slow"):
        return conftest.CACHE_DIR
    cache = tmp_path_factory.mktemp("cache")
    monkeypatch.setattr(conftest, "CACHE_DIR", cache)
    return cache


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def _pack(root: Path, relpath: str, payload: bytes) -> None:
    packed = root / (relpath + ".xz")
    packed.parent.mkdir(parents=True, exist_ok=True)
    packed.write_bytes(lzma.compress(payload))


def test_raw_file_resolves_to_its_path_in_the_data_root(
    tmp_path, monkeypatch, real_data_file
):
    raw = tmp_path / RELPATH
    raw.parent.mkdir(parents=True)
    raw.write_bytes(os.urandom(64))
    monkeypatch.setenv("MAFM_TEST_DATA", str(tmp_path))

    assert Path(real_data_file(RELPATH)) == raw


def test_raw_file_wins_over_packed_copy(tmp_path, monkeypatch, real_data_file):
    raw = tmp_path / RELPATH
    raw.parent.mkdir(parents=True)
    raw.write_bytes(os.urandom(64))
    _pack(tmp_path, RELPATH, os.urandom(64))
    monkeypatch.setenv("MAFM_TEST_DATA", str(tmp_path))

    assert Path(real_data_file(RELPATH)) == raw


def test_packed_file_resolves_to_unpacked_copy_outside_the_data_root(
    tmp_path, monkeypatch, real_data_file
):
    payload = os.urandom(4096)
    _pack(tmp_path, RELPATH, payload)
    before = _snapshot(tmp_path)
    monkeypatch.setenv("MAFM_TEST_DATA", str(tmp_path))

    path = Path(real_data_file(RELPATH))

    assert path.read_bytes() == payload
    assert not path.resolve().is_relative_to(tmp_path.resolve())
    assert _snapshot(tmp_path) == before


def test_second_request_for_packed_file_reuses_the_unpacked_copy(
    tmp_path, monkeypatch, real_data_file
):
    payload = os.urandom(4096)
    _pack(tmp_path, RELPATH, payload)
    monkeypatch.setenv("MAFM_TEST_DATA", str(tmp_path))

    first = Path(real_data_file(RELPATH))
    first_stat = first.stat()
    second = Path(real_data_file(RELPATH))

    assert second == first
    assert second.stat().st_ino == first_stat.st_ino
    assert second.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert second.read_bytes() == payload


def test_packed_files_at_the_same_path_in_two_roots_keep_their_own_bytes(
    tmp_path, monkeypatch, real_data_file
):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    payload_a = os.urandom(4096)
    payload_b = os.urandom(4096)
    _pack(root_a, RELPATH, payload_a)
    _pack(root_b, RELPATH, payload_b)

    monkeypatch.setenv("MAFM_TEST_DATA", str(root_a))
    path_a = Path(real_data_file(RELPATH))
    monkeypatch.setenv("MAFM_TEST_DATA", str(root_b))
    path_b = Path(real_data_file(RELPATH))

    assert path_a.read_bytes() == payload_a
    assert path_b.read_bytes() == payload_b


def test_unpacked_copy_lands_in_the_cache_dir(tmp_path, monkeypatch, private_cache):
    _pack(tmp_path, RELPATH, os.urandom(64))
    monkeypatch.setenv("MAFM_TEST_DATA", str(tmp_path))

    path = conftest.resolve_real_data_file(RELPATH)

    assert path.resolve().is_relative_to(private_cache.resolve())


def test_unpacking_prunes_stale_cache_entries_and_keeps_fresh_ones(
    tmp_path, monkeypatch, private_cache
):
    stale = private_cache / "stale"
    fresh = private_cache / "fresh"
    for entry in (stale, fresh):
        (entry / "f").mkdir(parents=True)
    old = time.time() - conftest.CACHE_MAX_AGE_S - 3600
    os.utime(stale, (old, old))
    _pack(tmp_path, RELPATH, os.urandom(64))
    monkeypatch.setenv("MAFM_TEST_DATA", str(tmp_path))

    conftest.resolve_real_data_file(RELPATH)

    assert not stale.exists()
    assert fresh.exists()


def test_missing_file_skips_naming_the_file_and_the_env_var(
    tmp_path, monkeypatch, real_data_file
):
    monkeypatch.setenv("MAFM_TEST_DATA", str(tmp_path))

    with pytest.raises(pytest.skip.Exception) as excinfo:
        real_data_file(RELPATH)

    assert RELPATH in str(excinfo.value)
    assert "MAFM_TEST_DATA" in str(excinfo.value)


@pytest.mark.slow
def test_example_data_corpus_resolves_packed_and_raw_files(monkeypatch, real_data_file):
    monkeypatch.delenv("MAFM_TEST_DATA", raising=False)
    packed_relpath = "test_cases/PS_PCL_SAA/QNM/maxscanrate.000.pfc"
    raw_relpath = "test_cases/DOE4_20nN_80nm.0_00012.spm"

    sums = Path(real_data_file("test_cases/SHA256SUMS.txt")).read_text()
    expected = {
        "test_cases/" + line.split()[-1]: line.split()[0]
        for line in sums.splitlines()
        if line and not line.startswith("#")
    }
    packed = Path(real_data_file(packed_relpath))
    raw = Path(real_data_file(raw_relpath))

    assert hashlib.sha256(packed.read_bytes()).hexdigest() == expected[packed_relpath]
    assert raw.samefile(REPO_ROOT / "example.data" / raw_relpath)
    assert raw.read_bytes().startswith(b"\\*Force file list")
