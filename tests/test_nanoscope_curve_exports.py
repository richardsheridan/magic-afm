"""Nanoscope curves read as Nanoscope Analysis's single-curve binary exports of them
show them."""

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

# Each export is one curve of its source file: a Nanoscope header, then one section of
# samples per channel, each the extend half stored near end first, then the retract
# half. The readers refuse these files (no Ciao image list), so the test decodes them
# with its own header parser. Export ForceCurveIndex N is the reader's pixel
# divmod(N, columns). Nanoscope Analysis rewrites each code c of a channel whose hard
# scale is per 16-bit LSB (Bytes/pixel 2) as floor(c x 32767 / 32768), so positive codes
# drop by one and the rest stay; a channel per 32-bit LSB (Bytes/pixel 4) is copied
# unchanged. Bytes/pixel is the width of that LSB, not of a stored sample: the A1
# Deflection Error sections hold 4-byte samples.

import re
from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile

A1 = "A1_OCSOH-PMMA21k_300421.0_300425.spm"
# reader (3, 4), and (12, 3), whose Height Sensor extend is invalid
A1_EXPORTS = [
    f"{A1}-6616_ForceCurveIndex_100.spm",
    f"{A1}-6616_ForceCurveIndex_387.spm",
]
A1_INVALID_EXPORT = A1_EXPORTS[1]
PFC = "PS_PCL_SAA/QNM/onekhz.001.pfc"
# reader (9, 8)
PFC_EXPORT = "onekhz.001.pfc-8B8F_ForceCurveIndex_296.spm"
# the stored index of the sample a .pfc source holds as -32768 to mark the extend half
PFC_MARKER = 127
INT32_MIN = np.iinfo(np.int32).min


def read_header(path):
    """The file's bytes, and its header sections in file order as (name, keys) pairs."""
    data = path.read_bytes()
    text = data[: data.index(b"\x1a")].decode("windows-1252")
    sections = []
    for line in text.split("\r\n"):
        if line.startswith("\\*"):
            sections.append((line[2:], {}))
        elif line.startswith("\\"):
            key, _, value = line[1:].partition(": ")
            sections[-1][1][key] = value
    return data, sections


def export_channel(path, name):
    """The codes of the export's first section called name, in time order, its nm per
    code by its own header, and its Bytes/pixel."""
    data, sections = read_header(path)
    (scan_list,) = [keys for n, keys in sections if n == "Ciao scan list"]
    keys = next(
        keys
        for n, keys in sections
        if n == "Ciao force image list" and keys["@4:Image Data"].split('"')[1] == name
    )
    n_ext, n_ret = map(int, keys["Samps/line"].split())
    width, rest = divmod(int(keys["Data length"]), n_ext + n_ret)
    assert rest == 0
    codes = np.frombuffer(
        data, f"<i{width}", n_ext + n_ret, int(keys["Data offset"])
    ).astype(np.int64)
    # reversed, the stored extend runs forward in time
    codes = np.concatenate([codes[:n_ext][::-1], codes[n_ext:]])

    scale = re.fullmatch(
        r"V \[Sens\. (\w+)\] \((\S+) V/LSB\) \S+ V", keys["@4:Z scale"]
    )
    assert scale is not None
    volts, sens, unit = scan_list[f"@Sens. {scale[1]}"].split()
    assert (volts, unit) == ("V", "nm/V")
    return codes, float(scale[2]) * float(sens), int(keys["Bytes/pixel"])


def reader_curve(real_data_file, source, export_name):
    """The reader's z and deflection in nm, in time order, at the export's pixel."""
    m = re.fullmatch(r".*_ForceCurveIndex_(\d+)\.spm", export_name)
    assert m is not None
    with open_fvfile(Path(real_data_file(source))) as fvfile:
        (volume,) = fvfile.volumes
        (z_ext, z_ret), (d_ext, d_ret) = volume.get_curve(
            *divmod(int(m[1]), volume.shape[1])
        )
    # extend near end last and retract near end first both run forward in time
    z = np.concatenate([z_ext, z_ret]).astype(np.float64)
    d = np.concatenate([d_ext, d_ret]).astype(np.float64)
    return z, d


def reader_codes(nm, nm_per_code):
    """nm as whole codes of nm_per_code, checking that they are whole."""
    codes = nm / nm_per_code
    np.testing.assert_allclose(codes, np.round(codes), rtol=0, atol=1e-3)
    return np.round(codes).astype(np.int64)


def vendor_rescale(codes):
    """Nanoscope Analysis's rewrite of the codes of a channel per 16-bit LSB."""
    return codes * 32767 // 32768


@pytest.mark.slow
@pytest.mark.parametrize("export_name", A1_EXPORTS)
def test_a1_deflection_matches_the_nanoscope_analysis_curve_export(
    real_data_file, export_name
):
    """The export's first Deflection Error holds the reader's deflection codes as
    Nanoscope Analysis rewrites them, on every sample."""
    export, nm_per_code, lsb_bytes = export_channel(
        Path(real_data_file(export_name)), "Deflection Error"
    )
    _, d = reader_curve(real_data_file, A1, export_name)

    assert lsb_bytes == 2
    np.testing.assert_array_equal(vendor_rescale(reader_codes(d, nm_per_code)), export)


@pytest.mark.slow
@pytest.mark.parametrize("export_name", A1_EXPORTS)
def test_a1_height_sensor_matches_the_nanoscope_analysis_curve_export(
    real_data_file, export_name
):
    """Where the export's Height Sensor holds a number, the reader's z is that number
    decoded by the export's own header."""
    export, nm_per_code, lsb_bytes = export_channel(
        Path(real_data_file(export_name)), "Height Sensor"
    )
    z, _ = reader_curve(real_data_file, A1, export_name)
    valid = export != INT32_MIN

    assert lsb_bytes == 4
    # the reader's float32 z holds these codes to 6.5e-8 of themselves; a code is 2e-8
    np.testing.assert_allclose(z[valid], export[valid] * nm_per_code, rtol=1.2e-7)


@pytest.mark.slow
def test_a1_z_is_nan_where_the_nanoscope_analysis_export_holds_int32_min(
    real_data_file,
):
    """The reader's z is NaN exactly where the export's Height Sensor holds INT32_MIN,
    and its deflection is finite there."""
    export, _, _ = export_channel(
        Path(real_data_file(A1_INVALID_EXPORT)), "Height Sensor"
    )
    z, d = reader_curve(real_data_file, A1, A1_INVALID_EXPORT)
    invalid = export == INT32_MIN

    assert invalid.any()
    assert np.isnan(z[invalid]).all()
    np.testing.assert_array_equal(np.isfinite(z), ~invalid)
    assert np.isfinite(d[invalid]).all()


@pytest.mark.slow
def test_pfc_deflection_matches_the_nanoscope_analysis_curve_export(real_data_file):
    """At exactly one cyclic rotation of the export's record, it holds the reader's
    deflection codes as Nanoscope Analysis rewrites them, on every sample but the
    marker."""
    export, nm_per_code, lsb_bytes = export_channel(
        Path(real_data_file(PFC_EXPORT)), "Deflection Error"
    )
    _, d = reader_curve(real_data_file, PFC, PFC_EXPORT)
    codes = reader_codes(d, nm_per_code)
    expected = vendor_rescale(codes)
    n = len(export)
    assert lsb_bytes == 2
    assert len(codes) == n

    # in time order the stored extend is reversed, so the marker comes first
    marker = n // 2 - 1 - PFC_MARKER
    # the reader rotates the record by its sync distance; the test searches for it
    matches = []
    for offset in range(n):
        rotated = np.roll(export, -offset)
        at = (marker - offset) % n
        others = np.arange(n) != at
        if np.array_equal(expected[others], rotated[others]):
            matches.append((at, rotated[at]))
    assert len(matches) == 1
    ((at, vendor_fill),) = matches
    # vendor: mean of the marker's neighbours; reader: a copy of one; here they agree
    assert abs(codes[at] - vendor_fill) <= 1
