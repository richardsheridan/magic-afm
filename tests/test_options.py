"""Options JSON reading: bad input is a ValueError naming the key and a CLI usage
error, and the JSONs the released betas wrote still load."""

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

import io
from pathlib import Path

import pytest
from click.testing import CliRunner

from magic_afm._options import TraceChoice, load_options
from magic_afm.calculation import FitFix, FitMode
from magic_afm.cli._impl import main

OLD_JSON_DIR = Path(__file__).parent / "fixtures" / "options_json"

# (key the message must name, options JSON text)
BAD_VALUE_CASES = {
    "fit_mode_bad_name": ("fit_mode", '{"fit_mode": "NOPE"}'),
    "trace_bad_name": ("trace", '{"trace": "NOPE"}'),
    "radius_null": ("radius", '{"radius": null}'),
    "fit_mode_null": ("fit_mode", '{"fit_mode": null}'),
    "radius_list": ("radius", '{"radius": [1]}'),
    "radius_bad_string": ("radius", '{"radius": "abc"}'),
    "fit_fix_bad_string": ("fit_fix", '{"fit_fix": "x"}'),
    "unknown_key": ("bogus", '{"bogus": 1}'),
}

# options JSON text with no key to name
BAD_SHAPE_CASES = {
    "malformed": "{bad",
    "top_level_list": "[1, 2]",
    "top_level_null": "null",
}
NOT_OBJECT_CASES = {"top_level_list", "top_level_null"}


@pytest.mark.parametrize(
    ("key", "text"), BAD_VALUE_CASES.values(), ids=BAD_VALUE_CASES.keys()
)
def test_load_options_bad_value_raises_valueerror_naming_key(key, text):
    with pytest.raises(ValueError) as excinfo:
        load_options(io.StringIO(text))
    assert key in str(excinfo.value)


@pytest.mark.parametrize("case", BAD_SHAPE_CASES)
def test_load_options_bad_shape_raises_valueerror(case):
    with pytest.raises(ValueError) as excinfo:
        load_options(io.StringIO(BAD_SHAPE_CASES[case]))
    if case in NOT_OBJECT_CASES:
        assert "object" in str(excinfo.value)


def _run_cli_with_options_json(tmp_path, text):
    options_json = tmp_path / "options.json"
    options_json.write_text(text)
    # exists with a known suffix, so only the options JSON can be rejected
    data_file = tmp_path / "curves.ARDF"
    data_file.touch()
    return CliRunner().invoke(
        main, ["--options-json", str(options_json), str(data_file)]
    )


def _error_lines(output):
    return [line for line in output.splitlines() if line.startswith("Error:")]


@pytest.mark.parametrize(
    ("key", "text"), BAD_VALUE_CASES.values(), ids=BAD_VALUE_CASES.keys()
)
def test_cli_bad_options_json_value_is_usage_error_naming_key(tmp_path, key, text):
    result = _run_cli_with_options_json(tmp_path, text)
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    error_lines = _error_lines(result.output)
    assert len(error_lines) == 1, result.output
    assert key in error_lines[0]


@pytest.mark.parametrize("case", BAD_SHAPE_CASES)
def test_cli_bad_options_json_shape_is_usage_error(tmp_path, case):
    result = _run_cli_with_options_json(tmp_path, BAD_SHAPE_CASES[case])
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert len(_error_lines(result.output)) == 1, result.output


# What today's load_options decodes from the JSONs the betas wrote (T3a, see
# tests/fixtures/options_json/README.md). These pin current behaviour,
# including the read-compat findings left for A3 and B2c:
# - fit_fix is an int in every fixture (123, FitFix.DEFAULTS).
# - trace is an int from the GUI (1) but a name from the CLI ("TRACE").
# - the 1.0.0b2 CLI default M is 1e9 (meant as Pa) and now reads as 1e9 GPa;
#   the 1.0.0b2 GUI wrote 10 ** (6 - 9), already GPa; 1.0.0b3 wrote 1.0 GPa.
# - the 1.0.0b2 CLI wrote no k_sens key; the GUI and 1.0.0b3 did.
_FIT_DEFAULTS = dict(
    radius=20.0,
    tau=0.0,
    lj_scale=2.0,
    vd=0.0,
    li_per=0.0,
    li_amp=0.0,
    drag=0.0,
    fit_fix=FitFix.DEFAULTS,
    fit_mode=FitMode.RETRACT,
)
_PFC = dict(k=0.46, defl_sens=35.0, sync_dist=69.6625)
OLD_JSON_VALUES = {
    "1.0.0b2-cli": {**_FIT_DEFAULTS, **_PFC, "trace": None, "M": 1e9},
    "1.0.0b2-cli-trace": {
        **_FIT_DEFAULTS,
        **_PFC,
        "trace": TraceChoice.TRACE,
        "M": 1e9,
    },
    "1.0.0b2-gui-pfc": {
        **_FIT_DEFAULTS,
        **_PFC,
        "trace": None,
        "M": 0.001,
        "k_sens": True,
    },
    "1.0.0b2-gui-ardf": {
        **_FIT_DEFAULTS,
        "k": 0.462,
        "defl_sens": 55.471000000000004,
        "sync_dist": None,
        "trace": TraceChoice.TRACE,
        "M": 0.001,
        "k_sens": True,
    },
    "1.0.0b3-cli": {**_FIT_DEFAULTS, **_PFC, "trace": None, "M": 1.0, "k_sens": True},
}
ENUM_TYPES = {"trace": TraceChoice, "fit_fix": FitFix, "fit_mode": FitMode}


def test_every_old_json_fixture_is_checked():
    assert {p.stem for p in OLD_JSON_DIR.glob("*.json")} == OLD_JSON_VALUES.keys()


@pytest.mark.parametrize("name", OLD_JSON_VALUES)
def test_load_options_reads_json_written_by_released_beta(name):
    with (OLD_JSON_DIR / (name + ".json")).open() as fp:
        options = load_options(fp)
    assert options == OLD_JSON_VALUES[name]
    for key, enum_type in ENUM_TYPES.items():
        if options[key] is not None:
            assert type(options[key]) is enum_type, key
