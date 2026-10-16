"""Malformed and unsupported files raise FileFormatError, with or without asserts."""

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

# Each case builds a good synthetic file and corrupts one field of it, picked so that a
# reader without its check would accept the file under python -O or fail with an error that is
# not a format error. ARDF fields are corrupted with their section's CRC rewritten, so
# the CRC check passes and the field's own check is the one that must fire. Every case
# is read twice: in this process, and in a python -O subprocess where asserts vanish.

import json
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.ardf import HEADER, VCHN, VDEF, build_ardf
from synthetic_files.arh5 import build_arh5_force_map
from synthetic_files.nanoscope import PFC_HEADER_BYTES, build_pfc

from magic_afm import data_readers

REPO_ROOT = Path(__file__).resolve().parent.parent
MAXSCANRATE_RELPATH = "test_cases/PS_PCL_SAA/QNM/maxscanrate.000.pfc"

# Runs in a fresh interpreter under -O: argv[1] is this directory, argv[2] the file.
# The last line of stdout is the JSON of what reading the file raised.
OPTIMIZED_READ_SCRIPT = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from test_format_errors import read_outcome

print(json.dumps({"optimize": sys.flags.optimize, **read_outcome(sys.argv[2])}))
"""


def read_everything(path: Path) -> None:
    """Parse path as the CLI does, then get every curve and image it holds."""
    with open_fvfile(Path(path)) as fvfile:
        for volume in fvfile.volumes:
            volume.get_curve(0, 0)
            volume.get_all_curves()
        for image in fvfile.images.values():
            image.get_image()


def read_outcome(path) -> dict:
    """The exception class names (its MRO) and message of read_everything, if any."""
    try:
        read_everything(Path(path))
    except Exception as exc:
        return {"mro": [cls.__name__ for cls in type(exc).__mro__], "message": str(exc)}
    return {"mro": None, "message": None}


def assert_format_error(path: Path, match: str) -> None:
    """Reading path raises FileFormatError matching match, here and under -O."""
    proc = subprocess.run(
        [sys.executable, "-O", "-c", OPTIMIZED_READ_SCRIPT, str(Path(__file__).parent)]
        + [str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    outcome = json.loads(proc.stdout.strip().splitlines()[-1])
    assert outcome["optimize"] >= 1
    assert outcome["mro"] is not None, "parsed without error under python -O"
    assert "FileFormatError" in outcome["mro"], outcome
    assert re.search(match, outcome["message"]), outcome

    with pytest.raises(data_readers.FileFormatError, match=match):
        read_everything(path)


def _recrc(data: bytearray, offset: int) -> None:
    """Rewrite the CRC of the ARDF section at offset to match its bytes."""
    _, size, _, _ = HEADER.unpack_from(data, offset)
    struct.pack_into("<L", data, offset, zlib.crc32(data[offset + 4 : offset + size]))


def _section(data: bytes, name: bytes) -> int:
    """Offset of the first ARDF section with this name."""
    offset = data.find(name) - 8
    assert offset >= 0 and HEADER.unpack_from(data, offset)[2] == name
    return offset


def _edit_nanoscope_header(path: Path, edit) -> None:
    """Apply edit(lines) to the header lines of a capture, keeping its size."""
    data = path.read_bytes()
    end = data.index(b"\x1a")
    lines = data[:end].decode("windows-1252").split("\r\n")
    text = "\r\n".join(edit(lines)).encode("windows-1252")
    assert len(text) < PFC_HEADER_BYTES
    header = text + b"\x1a" + bytes(PFC_HEADER_BYTES - len(text) - 1)
    path.write_bytes(header + data[PFC_HEADER_BYTES:])


def _set_key(lines, section: str, key: str, value: str) -> list[str]:
    """Header lines with key of the first section of this name set to value."""
    out, current, done = [], None, False
    for line in lines:
        if line.startswith("\\*"):
            current = line[2:]
        elif current == section and not done and line.startswith(f"\\{key}:"):
            line = f"\\{key}: {value}"
            done = True
        out.append(line)
    assert done, (section, key)
    return out


def test_file_format_error_is_a_value_error():
    assert issubclass(data_readers.FileFormatError, ValueError)


def test_ardf_with_a_channel_not_in_metres_raises_format_error(tmp_path):
    path = build_ardf(tmp_path).path
    data = bytearray(path.read_bytes())
    offset = _section(data, b"VCHN")
    name, _ = VCHN.unpack_from(data, offset + HEADER.size)
    VCHN.pack_into(data, offset + HEADER.size, name, b"V")
    _recrc(data, offset)
    path.write_bytes(data)
    assert_format_error(path, r"(?i)unit")


def test_ardf_with_a_wrong_segment_count_raises_format_error(tmp_path):
    path = build_ardf(tmp_path).path
    data = bytearray(path.read_bytes())
    offset = _section(data, b"VDEF")
    *fields, nseg = VDEF.unpack_from(data, offset + HEADER.size)
    VDEF.pack_into(data, offset + HEADER.size, *fields, nseg + 1)
    _recrc(data, offset)
    path.write_bytes(data)
    assert_format_error(path, r"(?i)segment")


def test_ardf_with_a_wrong_table_stride_raises_format_error(tmp_path):
    path = build_ardf(tmp_path).path
    data = bytearray(path.read_bytes())
    offset = _section(data, b"FTOC")
    size, slots, stride = struct.unpack_from("<QLL", data, offset + HEADER.size)
    struct.pack_into("<QLL", data, offset + HEADER.size, size, slots, stride + 8)
    _recrc(data, offset)
    path.write_bytes(data)
    assert_format_error(path, r"(?i)stride")


def test_arh5_force_map_with_a_dwell_segment_raises_format_error(tmp_path):
    path = build_arh5_force_map(tmp_path).path
    with h5py.File(path, "r+") as f:
        group = f["ForceMap/0"]
        ext, ret, away = np.moveaxis(group["Segments"][...], -1, 0)
        del group["Segments"]
        group.create_dataset("Segments", data=np.stack([ext, ext, ret, away], -1))
    assert_format_error(path, r"(?i)segment")


def test_nanoscope_pft_freq_not_in_khz_raises_format_error(tmp_path):
    path = build_pfc(tmp_path).path
    _edit_nanoscope_header(
        path, lambda lines: _set_key(lines, "Ciao scan list", "PFT Freq", "1000 Hz")
    )
    assert_format_error(path, r"PFT Freq")


def test_nanoscope_header_line_without_backslash_raises_format_error(tmp_path):
    path = build_pfc(tmp_path).path

    def insert_stray_line(lines):
        i = lines.index("\\*Equipment list")
        return lines[: i + 1] + ["Stray: 1"] + lines[i + 1 :]

    _edit_nanoscope_header(path, insert_stray_line)
    assert_format_error(path, re.escape("Stray: 1"))


def test_nanoscope_skipped_point_capture_raises_format_error(tmp_path):
    rows, cols = 4, 4
    path = build_pfc(tmp_path, (rows, cols)).path
    data = path.read_bytes()
    force_length = None

    def skip_points(lines):
        nonlocal force_length
        section = None
        for line in lines:
            if line.startswith("\\*"):
                section = line[2:]
            elif section == "Ciao force image list" and line.startswith(
                "\\Data length:"
            ):
                force_length = int(line.split(":")[1])
        lines = _set_key(lines, "Ciao force list", "force/line", f"{cols // 2} 0")
        return _set_key(
            lines, "Ciao force image list", "Data length", str(force_length // 4)
        )

    _edit_nanoscope_header(path, skip_points)
    # every other point of every other line: a quarter of the curves
    path.write_bytes(path.read_bytes()[: len(data) - force_length * 3 // 4])
    assert_format_error(path, r"(?i)skip|force/line")


@pytest.mark.slow
def test_real_nanoscope_skipped_point_capture_raises_format_error(real_data_file):
    assert_format_error(real_data_file(MAXSCANRATE_RELPATH), r"(?i)skip|force/line")
