"""Shared pytest fixtures for magic_afm tests."""

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
import shutil
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(tempfile.gettempdir()) / "magic-afm-test-data"


def _data_root() -> Path:
    return Path(os.environ.get("MAFM_TEST_DATA") or REPO_ROOT / "example.data")


def _unpack_cached(packed: Path, relpath: str) -> Path:
    src = packed.resolve()
    st = src.stat()
    key = hashlib.sha256(f"{src}\0{st.st_size}\0{st.st_mtime_ns}".encode())
    dest = CACHE_DIR / key.hexdigest()[:16] / relpath
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, lzma.open(src) as inp:
            shutil.copyfileobj(inp, out)
        os.replace(tmp, dest)
    except BaseException:
        os.unlink(tmp)
        raise
    return dest


def resolve_real_data_file(relpath: str) -> Path:
    """Return a readable path for relpath in the data root, unpacking .xz if needed."""
    root = _data_root()
    raw = root / relpath
    if raw.is_file():
        return raw
    packed = root / (relpath + ".xz")
    if packed.is_file():
        return _unpack_cached(packed, relpath)
    pytest.skip(f"real data file {relpath} not found under {root}; set MAFM_TEST_DATA")


@pytest.fixture
def real_data_file():
    return resolve_real_data_file
