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
# reader without its check would accept the file under python -O or fail with an error
# that is not a format error. ARDF fields are corrupted with their section's CRC
# rewritten, so the CRC check passes and the field's own check is the one that must
# fire. Every case is read twice: in this process, and in a python -O subprocess where
# asserts vanish.

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
from synthetic_files.nanoscope import (
    PFC_HEADER_BYTES,
    SPM_HEADER_BYTES,
    build_pfc,
    build_spm,
    edit_nanoscope_header,
)
from synthetic_files.nid import build_nid

from magic_afm import data_readers

REPO_ROOT = Path(__file__).resolve().parent.parent
MAXSCANRATE_RELPATH = "test_cases/PS_PCL_SAA/QNM/maxscanrate.000.pfc"

# Runs in a fresh interpreter under -O: argv[1] is this directory, argv[2] the file. The
# last line of stdout is the JSON of what reading the file raised.
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


def _replace_in_key(lines, section: str, key: str, old: str, new: str) -> list[str]:
    """Header lines with old replaced by new in key of the first such section."""
    value = next(
        line.split(": ", 1)[1]
        for line in _section_lines(lines, section)
        if line.startswith(f"\\{key}:")
    )
    assert old in value, (key, value)
    return _set_key(lines, section, key, value.replace(old, new))


def _section_lines(lines, section: str) -> list[str]:
    """The lines of the first section of this name."""
    starts = [i for i, line in enumerate(lines) if line == f"\\*{section}"]
    assert starts, section
    out = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("\\*"):
            break
        out.append(line)
    return out


def _edit_nid_header(path: Path, edit) -> None:
    """Apply edit(lines) to the header lines of a NID file, before its #! mark."""
    data = path.read_bytes()
    end = data.index(b"#!")
    lines = data[:end].decode("utf-8").split("\r\n")
    path.write_bytes("\r\n".join(edit(lines)).encode("utf-8") + data[end:])


def _set_nid_curve_unit(lines, channel: str, unit: str) -> list[str]:
    """Header lines with the Dim2Unit of the channel's spectroscopy curves set."""
    out, frame, name, done = [], None, None, 0
    for line in lines:
        if line.startswith("["):
            frame = name = None
        elif line.startswith("Frame="):
            frame = line.split("=", 1)[1]
        elif line.startswith("Dim2Name="):
            name = line.split("=", 1)[1]
        elif line.startswith("Dim2Unit=") and name == channel:
            if frame in ("Spec forward", "Spec backward"):
                line = f"Dim2Unit={unit}"
                done += 1
        out.append(line)
    assert done == 2, (channel, done)
    return out


def _set_nid_spring_constant_unit(lines, unit: str) -> list[str]:
    """Header lines with the unit of the cantilever's Prop0 set."""
    found = [i for i, line in enumerate(lines) if line.startswith("Prop0=")]
    assert len(found) == 1, found
    value = lines[found[0]].split("]*[")[0]
    return lines[: found[0]] + [f"{value}]*[{unit}]"] + lines[found[0] + 1 :]


def _set_nid_map_grid(lines, rows: int, cols: int) -> list[str]:
    """Header lines with the column and row counts of the Map0 row set."""
    found = [i for i, line in enumerate(lines) if line.startswith("Map0=")]
    assert len(found) == 1, found
    fields = lines[found[0]].split("=", 1)[1].split(";")
    fields[4:6] = [str(cols), str(rows)]
    return lines[: found[0]] + ["Map0=" + ";".join(fields)] + lines[found[0] + 1 :]


def test_file_format_error_is_a_value_error():
    assert issubclass(data_readers.FileFormatError, ValueError)


def test_ardf_with_a_channel_in_an_unknown_unit_raises_format_error(tmp_path):
    path = build_ardf(tmp_path).path
    data = bytearray(path.read_bytes())
    offset = _section(data, b"VCHN")
    name, _ = VCHN.unpack_from(data, offset + HEADER.size)
    VCHN.pack_into(data, offset + HEADER.size, name, b"furlong")
    _recrc(data, offset)
    path.write_bytes(data)
    assert_format_error(path, "furlong")


@pytest.mark.parametrize(
    "channel, unit", [("Z-Axis Sensor", "furlong"), ("Deflection", "lbf")]
)
def test_nid_curves_in_an_unknown_unit_raise_format_error(tmp_path, channel, unit):
    path = build_nid(tmp_path).path
    _edit_nid_header(path, lambda lines: _set_nid_curve_unit(lines, channel, unit))
    assert_format_error(path, re.escape(unit))


def test_nid_spring_constant_in_an_unknown_unit_raises_format_error(tmp_path):
    path = build_nid(tmp_path).path
    _edit_nid_header(path, lambda lines: _set_nid_spring_constant_unit(lines, "lbf/in"))
    assert_format_error(path, re.escape("lbf/in"))


@pytest.mark.parametrize(
    "rows, cols",
    [(3, 1), (1, 4), (1, 1), (3, 0), (0, 4)],
    ids=["one-column", "one-row", "one-by-one", "no-columns", "no-rows"],
)
def test_nid_map_of_fewer_than_two_rows_or_columns_raises_format_error(
    tmp_path, rows, cols
):
    """A map without two points along each axis fails to parse, naming its grid."""
    path = build_nid(tmp_path, (3, 4)).path
    _edit_nid_header(path, lambda lines: _set_nid_map_grid(lines, rows, cols))
    with pytest.raises(data_readers.FileFormatError, match="grid"):
        with open_fvfile(path):
            pass
    assert_format_error(path, "grid")


@pytest.mark.parametrize(
    "build, size, key",
    [
        (build_pfc, PFC_HEADER_BYTES, "@Sens. DeflSens"),
        (build_spm, SPM_HEADER_BYTES, "@Sens. DeflSens"),
        (build_spm, SPM_HEADER_BYTES, "@Sens. ZsensSens"),
    ],
    ids=["pfc-DeflSens", "spm-DeflSens", "spm-ZsensSens"],
)
def test_nanoscope_curve_sensitivity_in_an_unknown_unit_raises_format_error(
    tmp_path, build, size, key
):
    path = build(tmp_path).path
    edit_nanoscope_header(
        path,
        lambda lines: _replace_in_key(lines, "Ciao scan list", key, "nm/V", "nA/V"),
        size,
    )
    assert_format_error(path, re.escape("nA/V"))


def test_nanoscope_curve_scale_in_an_unknown_unit_raises_format_error(tmp_path):
    path = build_pfc(tmp_path).path
    edit_nanoscope_header(
        path,
        lambda lines: _replace_in_key(
            lines, "Ciao force image list", "@4:Z scale", "V/LSB", "Arb/LSB"
        ),
    )
    assert_format_error(path, re.escape("Arb/LSB"))


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
    edit_nanoscope_header(
        path, lambda lines: _set_key(lines, "Ciao scan list", "PFT Freq", "1000 Hz")
    )
    assert_format_error(path, r"PFT Freq")


@pytest.mark.parametrize(
    "value",
    ["V (0.0000001164 kHz/LSB) 976.5625 Hz", "V (0.0000001164 kHz/LSB) 0 kHz"],
    ids=["not-khz", "zero"],
)
def test_pfc_drive_frequency_not_in_khz_or_zero_raises_format_error(tmp_path, value):
    path = build_pfc(tmp_path).path
    edit_nanoscope_header(
        path,
        lambda lines: _set_key(
            lines, "Ciao scan list", "@2:DriveFrequency3SoftHarmoniX", value
        ),
    )
    assert_format_error(path, "DriveFrequency3SoftHarmoniX")


def test_nanoscope_header_line_without_backslash_raises_format_error(tmp_path):
    path = build_pfc(tmp_path).path

    def insert_stray_line(lines):
        i = lines.index("\\*Equipment list")
        return lines[: i + 1] + ["Stray: 1"] + lines[i + 1 :]

    edit_nanoscope_header(path, insert_stray_line)
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

    edit_nanoscope_header(path, skip_points)
    # every other point of every other line: a quarter of the curves
    path.write_bytes(path.read_bytes()[: len(data) - force_length * 3 // 4])
    assert_format_error(path, r"(?i)skip|force/line")


@pytest.mark.slow
def test_real_nanoscope_skipped_point_capture_raises_format_error(real_data_file):
    assert_format_error(real_data_file(MAXSCANRATE_RELPATH), r"(?i)skip|force/line")


NANOSCOPE_BUILDS = [(build_pfc, PFC_HEADER_BYTES), (build_spm, SPM_HEADER_BYTES)]


@pytest.mark.parametrize(
    "build, size, length",
    [
        (build_pfc, PFC_HEADER_BYTES, 33),
        (build_pfc, PFC_HEADER_BYTES, 16),
        (build_pfc, PFC_HEADER_BYTES, 48),
        (build_pfc, PFC_HEADER_BYTES, 128),
        (build_pfc, PFC_HEADER_BYTES, 256),
        (build_spm, SPM_HEADER_BYTES, 65),
        (build_spm, SPM_HEADER_BYTES, 16),
        (build_spm, SPM_HEADER_BYTES, 48),
        (build_spm, SPM_HEADER_BYTES, 128),
        (build_spm, SPM_HEADER_BYTES, 256),
    ],
    ids=[
        "pfc-2-bytes-and-1-over",
        "pfc-1-byte",
        "pfc-3-bytes",
        "pfc-8-bytes",
        "pfc-16-bytes",
        "spm-4-bytes-and-1-over",
        "spm-1-byte",
        "spm-3-bytes",
        "spm-8-bytes",
        "spm-16-bytes",
    ],
)
def test_nanoscope_image_length_not_whole_samples_of_2_or_4_bytes_raises_format_error(
    tmp_path, build, size, length
):
    """A 4 x 4 image whose Data length is not 32 or 64 bytes fails to parse."""
    path = build(tmp_path, (4, 4)).path
    edit_nanoscope_header(
        path,
        lambda lines: _set_key(lines, "Ciao image list", "Data length", str(length)),
        size,
    )
    with pytest.raises(data_readers.FileFormatError, match="Data length"):
        with open_fvfile(path):
            pass
    assert_format_error(path, "Data length")


# id: section, key and its zero value in each format whose reader takes the key; a
# force volume takes its grid from the force list, a capture from the scan list
NANOSCOPE_ZERO_CASES = {
    "image-Number-of-lines": (
        "Ciao image list",
        "Number of lines",
        {"pfc": "0", "spm": "0"},
    ),
    "image-Samps-line": ("Ciao image list", "Samps/line", {"pfc": "0", "spm": "0"}),
    "scan-Samps-line": ("Ciao scan list", "Samps/line", {"pfc": "0"}),
    "scan-Lines": ("Ciao scan list", "Lines", {"pfc": "0"}),
    "PFT-Freq": ("Ciao scan list", "PFT Freq", {"pfc": "0 KHz", "spm": "0 kHz"}),
    "force-line": ("Ciao force list", "force/line", {"spm": "0"}),
    "FV-Lines": ("Ciao force list", "FV Lines", {"spm": "0"}),
    "force-image-Samps-line": (
        "Ciao force image list",
        "Samps/line",
        {"pfc": "0 0", "spm": "0 0"},
    ),
}
NANOSCOPE_BUILDS_BY_FORMAT = dict(zip(("pfc", "spm"), NANOSCOPE_BUILDS))


@pytest.mark.parametrize(
    "build, size, section, key, value",
    [
        pytest.param(build, size, section, key, values[fmt], id=f"{case}-{fmt}")
        for case, (section, key, values) in NANOSCOPE_ZERO_CASES.items()
        for fmt, (build, size) in NANOSCOPE_BUILDS_BY_FORMAT.items()
        if fmt in values
    ],
)
def test_nanoscope_zero_count_or_rate_raises_format_error_naming_it(
    tmp_path, build, size, section, key, value
):
    """A zero count or rate the reader divides by fails to parse, naming its key."""
    path = build(tmp_path).path
    edit_nanoscope_header(
        path, lambda lines: _set_key(lines, section, key, value), size
    )
    assert_format_error(path, re.escape(key))


@pytest.mark.parametrize("width", [3, 8])
def test_spm_force_length_not_2_or_4_byte_samples_raises_format_error_naming_grid(
    tmp_path, width
):
    """The message names the force list's FV Lines and force/line."""
    (rows, cols), points = (4, 4), 16
    path = build_spm(tmp_path, (rows, cols), points).path
    length = rows * cols * 2 * points * width
    edit_nanoscope_header(
        path,
        lambda lines: _set_key(
            lines, "Ciao force image list", "Data length", str(length)
        ),
        SPM_HEADER_BYTES,
    )
    assert_format_error(path, r"(?s)(?=.*FV Lines)(?=.*force/line)")


@pytest.mark.parametrize(
    "force_sections",
    [("peak force error", "height sensor"), ("height sensor",)],
    ids=["peak-force-error", "height-only"],
)
def test_spm_without_a_deflection_error_raises_format_error(tmp_path, force_sections):
    path = build_spm(tmp_path, force_sections=force_sections).path
    assert_format_error(path, "Deflection Error")


@pytest.mark.parametrize("sample_points", ["0", "-1"])
def test_pfc_sample_points_not_positive_raises_format_error(tmp_path, sample_points):
    path = build_pfc(tmp_path).path
    edit_nanoscope_header(
        path,
        lambda lines: _set_key(lines, "Ciao scan list", "Sample Points", sample_points),
    )
    assert_format_error(path, "Sample Points")


def test_pfc_zero_pft_freq_without_sample_points_raises_format_error(tmp_path):
    path = build_pfc(tmp_path).path

    def zero_rate_no_sample_points(lines):
        kept = [line for line in lines if not line.startswith("\\Sample Points:")]
        assert len(kept) == len(lines) - 1
        return _set_key(kept, "Ciao scan list", "PFT Freq", "0 KHz")

    edit_nanoscope_header(path, zero_rate_no_sample_points)
    assert_format_error(path, "PFT Freq")
