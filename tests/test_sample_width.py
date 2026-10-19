"""Each reader states the byte width of its file's numeric data from its header."""

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

import inspect
import json
import subprocess
import sys
import typing
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.ardf import build_ardf
from synthetic_files.arh5 import build_arh5_ffm, build_arh5_force_map
from synthetic_files.nanoscope import (
    PFC_HEADER_BYTES,
    SPM_HEADER_BYTES,
    build_pfc,
    build_spm,
    edit_nanoscope_header,
)
from synthetic_files.nid import build_nid

from magic_afm.data_readers import SUFFIX_FVFILE_MAP, FileFormatError, FVFile

REPO_ROOT = Path(__file__).resolve().parent.parent

# builder and the width in bytes it stores every numeric section at
WIDTH_BUILDERS = {
    "ardf-trace": (partial(build_ardf, capture="trace"), 4),
    "ardf-retrace": (partial(build_ardf, capture="retrace"), 4),
    "ardf-both": (partial(build_ardf, capture="both"), 4),
    "ardf-slow-map": (partial(build_ardf, slow_map=True, ext_points=24), 4),
    "arh5-ffm-trace": (partial(build_arh5_ffm, capture="trace"), 4),
    "arh5-ffm-both": (partial(build_arh5_ffm, capture="both"), 4),
    "arh5-force-map": (build_arh5_force_map, 4),
    "pfc": (build_pfc, 2),
    "pfc-phase-image": (partial(build_pfc, phase_image=True), 2),
    "spm": (build_spm, 4),
    "spm-phase-image": (partial(build_spm, phase_image=True), 4),
    "spm-stale-scan-list": (
        partial(build_spm, shape=(8, 8), points=16, scan_list_grid=(8, 16)),
        4,
    ),
    "spm-peak-force-error-first": (
        partial(
            build_spm,
            force_sections=("peak force error", "deflection", "height sensor"),
        ),
        4,
    ),
    "nid": (build_nid, 4),
    "nid-16-bit": (partial(build_nid, save_bits=16), 2),
}
BUFFER_CASES = [case for case in WIDTH_BUILDERS if not case.startswith("arh5")]

FILE_CLASSES = sorted(
    {cls for cls, _ in SUFFIX_FVFILE_MAP.values()}, key=lambda cls: cls.__name__
)


def sample_width(path: Path, suffix: str | None = None):
    """sample_width of the class for suffix, given path opened by its opener."""
    cls, opener = SUFFIX_FVFILE_MAP[suffix or path.suffix.lower()]
    with opener(path) as data:
        return cls.sample_width(data)


# Runs in a fresh interpreter under -O: argv[1] is the suffix whose class to ask,
# argv[2] the file. The last line of stdout is the JSON of what the call raised.
OPTIMIZED_SCRIPT = """
import json
import sys

from magic_afm.data_readers import SUFFIX_FVFILE_MAP

cls, opener = SUFFIX_FVFILE_MAP[sys.argv[1]]
try:
    with opener(sys.argv[2]) as data:
        cls.sample_width(data)
except Exception as exc:
    mro = [c.__name__ for c in type(exc).__mro__]
else:
    mro = None
print(json.dumps({"optimize": sys.flags.optimize, "mro": mro}))
"""


def assert_refused(path: Path, suffix: str | None = None) -> None:
    """sample_width raises FileFormatError on path, here and under python -O."""
    suffix = suffix or path.suffix.lower()
    with pytest.raises(FileFormatError):
        sample_width(path, suffix)
    proc = subprocess.run(
        [sys.executable, "-O", "-c", OPTIMIZED_SCRIPT, suffix, str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    outcome = json.loads(proc.stdout.strip().splitlines()[-1])
    assert outcome["optimize"] >= 1
    assert outcome["mro"] is not None, "no error under python -O"
    assert "FileFormatError" in outcome["mro"], outcome


def _rewidth(lines, section: str, stored: int, width: int) -> list[str]:
    """Header lines with the first section's Data length at width bytes a sample."""
    out, current, done = [], None, False
    for line in lines:
        if line.startswith("\\*"):
            current = line[2:]
        elif current == section and not done and line.startswith("\\Data length:"):
            length = int(line.split(":", 1)[1])
            assert length % stored == 0, (section, length, stored)
            line = f"\\Data length: {length // stored * width}"
            done = True
        out.append(line)
    assert done, section
    return out


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


def _drop_sections(lines, section: str) -> list[str]:
    """Header lines without any section of this name."""
    out, dropping, dropped = [], False, 0
    for line in lines:
        if line.startswith("\\*"):
            dropping = line == f"\\*{section}"
            dropped += dropping
        if not dropping:
            out.append(line)
    assert dropped, section
    return out


def _force_length(path: Path) -> int:
    """Data length of the first Ciao force image list of a Nanoscope file."""
    header = path.read_bytes().split(b"\x1a", 1)[0].decode("windows-1252")
    section = None
    for line in header.split("\r\n"):
        if line.startswith("\\*"):
            section = line[2:]
        elif section == "Ciao force image list" and line.startswith("\\Data length:"):
            return int(line.split(":", 1)[1])
    raise AssertionError("no force image list")


def _set_last_nid_save_bits(path: Path, bits: int) -> None:
    """Rewrite the SaveBits of the last dataset in a NID header."""
    data = path.read_bytes()
    end = data.index(b"#!")
    lines = data[:end].decode("utf-8").split("\r\n")
    last = max(i for i, line in enumerate(lines) if line.startswith("SaveBits="))
    lines[last] = f"SaveBits={bits}"
    path.write_bytes("\r\n".join(lines).encode("utf-8") + data[end:])


def _widen_first_image(path: Path) -> None:
    """Rewrite the first image of an ARH5 file as float64, keeping its attributes."""
    with h5py.File(path, "r+") as f:
        images = f["Image"]
        name = next(iter(images))
        values = images[name][()]
        attrs = dict(images[name].attrs)
        del images[name]
        images.create_dataset(name, data=values.astype(np.float64))
        images[name].attrs.update(attrs)


def assert_mixed(path: Path) -> None:
    """The reader accepts path, and sample_width says its widths are mixed."""
    with open_fvfile(path) as fvfile:
        assert fvfile.volumes
    assert sample_width(path) is None


@pytest.mark.parametrize("cls", FILE_CLASSES, ids=lambda cls: cls.__name__)
def test_every_file_class_has_a_sample_width_staticmethod(cls):
    assert isinstance(inspect.getattr_static(cls, "sample_width"), staticmethod)


def test_fvfile_protocol_declares_sample_width_returning_int_or_none():
    """FVFile declares sample_width as a staticmethod returning int | None."""
    member = inspect.getattr_static(FVFile, "sample_width")
    assert isinstance(member, staticmethod)
    assert typing.get_type_hints(FVFile.sample_width)["return"] == int | None


@pytest.mark.parametrize("case", WIDTH_BUILDERS)
def test_sample_width_is_the_width_the_builder_stores_its_data_at(tmp_path, case):
    """Given what the map's opener gives, sample_width is the stored itemsize."""
    builder, width = WIDTH_BUILDERS[case]
    path = builder(tmp_path).path
    with open_fvfile(path) as fvfile:
        assert fvfile.volumes
    got = sample_width(path)
    assert type(got) is int and got == width, got


@pytest.mark.parametrize("case", BUFFER_CASES)
def test_sample_width_reads_the_file_from_its_bytes(tmp_path, case):
    """The mmap formats take any buffer holding the whole file, such as bytes."""
    builder, width = WIDTH_BUILDERS[case]
    path = builder(tmp_path).path
    cls, _ = SUFFIX_FVFILE_MAP[path.suffix.lower()]
    got = cls.sample_width(path.read_bytes())
    assert type(got) is int and got == width, got


@pytest.mark.parametrize(
    "build, size, section, stored, width",
    [
        (build_spm, SPM_HEADER_BYTES, "Ciao image list", 4, 2),
        (build_pfc, PFC_HEADER_BYTES, "Ciao image list", 2, 4),
    ],
    ids=["spm-image-2", "pfc-image-4"],
)
def test_nanoscope_image_and_force_sections_of_unlike_width_are_mixed(
    tmp_path, build, size, section, stored, width
):
    path = build(tmp_path, (4, 4)).path
    edit_nanoscope_header(
        path, lambda lines: _rewidth(lines, section, stored, width), size
    )
    assert_mixed(path)


@pytest.mark.parametrize(
    "force_sections",
    [
        ("peak force error", "deflection", "height sensor"),
        ("height sensor", "deflection"),
        ("deflection", "height sensor"),
    ],
    ids=["unread-peak-force-error", "z", "d"],
)
def test_spm_force_section_of_unlike_width_is_mixed(tmp_path, force_sections):
    """Every force section counts, read or not, and z against d too."""
    path = build_spm(tmp_path, (4, 4), 16, force_sections=force_sections).path
    edit_nanoscope_header(
        path,
        lambda lines: _rewidth(lines, "Ciao force image list", 4, 2),
        SPM_HEADER_BYTES,
    )
    assert_mixed(path)


def test_nid_dataset_of_unlike_width_is_mixed(tmp_path):
    """Every dataset counts, the last scan frame as much as the curves."""
    path = build_nid(tmp_path).path
    _set_last_nid_save_bits(path, 16)
    assert_mixed(path)


@pytest.mark.parametrize("build", [build_arh5_ffm, build_arh5_force_map])
def test_arh5_dataset_of_unlike_width_is_mixed(tmp_path, build):
    path = build(tmp_path).path
    _widen_first_image(path)
    assert_mixed(path)


@pytest.mark.parametrize(
    "suffix, foreign",
    [
        (".ardf", build_nid),
        (".spm", build_ardf),
        (".pfc", build_nid),
        (".nid", build_pfc),
    ],
    ids=["ardf-given-nid", "spm-given-ardf", "pfc-given-nid", "nid-given-pfc"],
)
def test_sample_width_of_a_file_in_another_format_raises_format_error(
    tmp_path, suffix, foreign
):
    path = foreign(tmp_path).path
    cls, _ = SUFFIX_FVFILE_MAP[suffix]
    with pytest.raises(FileFormatError):
        cls.sample_width(path.read_bytes())
    assert_refused(path, suffix)


NANOSCOPE_SECTIONS = (
    "Ciao image list",
    "Ciao scan list",
    "Ciao force list",
    "Ciao force image list",
)


@pytest.mark.parametrize(
    "section", NANOSCOPE_SECTIONS, ids=[s.replace(" ", "-") for s in NANOSCOPE_SECTIONS]
)
@pytest.mark.parametrize(
    "build, size",
    [(build_pfc, PFC_HEADER_BYTES), (build_spm, SPM_HEADER_BYTES)],
    ids=["pfc", "spm"],
)
def test_nanoscope_without_a_header_section_sample_width_raises_format_error(
    tmp_path, build, size, section
):
    path = build(tmp_path).path
    edit_nanoscope_header(path, lambda lines: _drop_sections(lines, section), size)
    assert_refused(path)


@pytest.mark.parametrize(
    "build, size, section, stored, width",
    [
        (build_pfc, PFC_HEADER_BYTES, "Ciao force image list", 2, 3),
        (build_pfc, PFC_HEADER_BYTES, "Ciao force image list", 2, 8),
        (build_pfc, PFC_HEADER_BYTES, "Ciao image list", 2, 3),
        (build_spm, SPM_HEADER_BYTES, "Ciao force image list", 4, 3),
        (build_spm, SPM_HEADER_BYTES, "Ciao force image list", 4, 8),
        (build_spm, SPM_HEADER_BYTES, "Ciao image list", 4, 1),
    ],
    ids=["pfc-force-3", "pfc-force-8", "pfc-image-3", "spm-force-3", "spm-force-8"]
    + ["spm-image-1"],
)
def test_nanoscope_section_not_of_2_or_4_byte_samples_raises_format_error(
    tmp_path, build, size, section, stored, width
):
    path = build(tmp_path, (4, 4)).path
    edit_nanoscope_header(
        path, lambda lines: _rewidth(lines, section, stored, width), size
    )
    assert_refused(path)


def test_pfc_skipped_point_capture_sample_width_raises_format_error(tmp_path):
    """Force data a quarter of the scan list's grid: half a byte a sample."""
    rows, cols = 4, 4
    path = build_pfc(tmp_path, (rows, cols)).path
    size = path.stat().st_size
    force_length = _force_length(path)

    def skip_points(lines):
        lines = _set_key(lines, "Ciao force list", "force/line", f"{cols // 2} 0")
        return _set_key(
            lines, "Ciao force image list", "Data length", str(force_length // 4)
        )

    edit_nanoscope_header(path, skip_points)
    path.write_bytes(path.read_bytes()[: size - force_length * 3 // 4])
    assert_refused(path)


@pytest.mark.parametrize(
    "build, size",
    [(build_pfc, PFC_HEADER_BYTES), (build_spm, SPM_HEADER_BYTES)],
    ids=["pfc", "spm"],
)
def test_nanoscope_single_curve_on_a_map_header_raises_format_error(
    tmp_path, build, size
):
    """One curve's bytes against the header's grid is a fraction of a byte a sample."""
    rows, cols = 4, 4
    path = build(tmp_path, (rows, cols)).path
    one_curve = _force_length(path) // (rows * cols)
    edit_nanoscope_header(
        path,
        lambda lines: _set_key(
            lines, "Ciao force image list", "Data length", str(one_curve)
        ),
        size,
    )
    assert_refused(path)


REAL_WIDTHS = {
    "PDMS_SAA/ffm/both0001.ARDF": 4,
    "PDMS_SAA/ffm/hard0001.ARDF": 4,
    "PDMS_SAA/ffm/hard0003.ARDF": 4,
    "PDMS_SAA/ffm/onekhz0001.ARDF": 4,
    "PDMS_SAA/ffm/rtrace0001.ARDF": 4,
    "PDMS_SAA/ffm/twoxln0001.ARDF": 4,
    "PDMS_SAA/ffm/both0001.h5": 4,
    "PDMS_SAA/ffm/rtrace0001.h5": 4,
    "A1_OCSOH-PMMA21k_300421.0_300425.spm": 4,
    "DOE4_20nN_80nm.0_00012.spm": 4,
    "20251216_141819_PVDF2_Film_U_00002.nid": 4,
    "PS_PCL_SAA/QNM/onekhz.000.pfc": 2,
}


@pytest.mark.slow
@pytest.mark.parametrize("relpath", REAL_WIDTHS)
def test_real_file_sample_width(real_data_file, relpath):
    got = sample_width(real_data_file(f"test_cases/{relpath}"))
    assert type(got) is int and got == REAL_WIDTHS[relpath], got


@pytest.mark.slow
def test_real_skipped_point_capture_sample_width_raises_format_error(real_data_file):
    assert_refused(real_data_file("test_cases/PS_PCL_SAA/QNM/maxscanrate.000.pfc"))
