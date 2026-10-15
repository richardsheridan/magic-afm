"""A bad options JSON is a ValueError naming the key, and a CLI usage error."""

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

import pytest
from click.testing import CliRunner

from magic_afm._options import load_options
from magic_afm.cli._impl import main

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
