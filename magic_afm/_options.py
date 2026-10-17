"""MagicAFM Options

The single source of truth for the fit and preprocessing options JSON written
by the GUI and CLI.

Add new fit or preprocessing parameters to OPTIONS_JSON_SCHEMA here.

Values are stored in the units of the fit, so e.g. radius is in nm and M is in GPa.
"""

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

import enum
import json
import warnings

from .calculation import FitFix, FitMode

SCHEMA_VERSION = 1
# M is in GPa; an unversioned file above this probably holds a value meant as Pa
IMPLAUSIBLE_M = 1e4


class TraceChoice(enum.IntEnum):
    RETRACE = 0
    TRACE = 1
    # BOTH = 2
    # ALL = -1


def trace_choice(v):
    # name (current) or legacy int (GUI-written)
    return TraceChoice[v] if isinstance(v, str) else TraceChoice(v)


def fit_fix(v):
    # list of member names (current) or legacy int
    if isinstance(v, list):
        flags = FitFix(0)
        for name in v:
            flags |= FitFix[name]
        return flags
    return FitFix(v)


def strict_int(v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(v)
    return v


OPTIONS_JSON_SCHEMA = dict(
    k=float,
    defl_sens=float,
    sync_dist=float,
    trace=trace_choice,
    k_sens=bool,
    radius=float,
    M=float,
    tau=float,
    lj_scale=float,
    vd=float,
    li_per=float,
    li_amp=float,
    drag=float,
    fit_fix=fit_fix,
    fit_mode=FitMode.__getitem__,
    resample_npts=strict_int,
)

NULLABLE_FIELDS = {"k", "defl_sens", "sync_dist", "trace"}


def load_options(fp):
    """Read an options JSON from an open file into a dict of validated values.

    Any subset of the keys in OPTIONS_JSON_SCHEMA may be present."""
    options = json.load(fp)
    if not isinstance(options, dict):
        raise ValueError("options_json must be a JSON object")
    version = options.pop("schema_version", None)
    if version is not None and version != SCHEMA_VERSION:
        raise ValueError(f"Bad value {version!r} for 'schema_version' in options_json")
    for k, value in list(options.items()):
        if k in NULLABLE_FIELDS and value is None:
            continue
        try:
            validator = OPTIONS_JSON_SCHEMA[k]
        except KeyError:
            raise ValueError(f"Unknown key '{k}' in options_json") from None
        try:
            options[k] = validator(value)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Bad value {value!r} for '{k}' in options_json") from e
    if version is None and options.get("M", 0) > IMPLAUSIBLE_M:
        warnings.warn(
            f"Implausible M of {options['M']} GPa in unversioned options_json, "
            "perhaps written in Pa by an old version",
            stacklevel=2,
        )
    return options


def dump_options(options):
    """Convert a dict of options to the text of an options JSON.

    The keys must be exactly those of OPTIONS_JSON_SCHEMA, ensuring
    the CLI will be able to read it back in."""
    mismatched = options.keys() ^ OPTIONS_JSON_SCHEMA.keys()
    if mismatched:
        raise ValueError(f"Options do not match OPTIONS_JSON_SCHEMA: {mismatched}")
    options = {"schema_version": SCHEMA_VERSION, **options}
    options = {k: options[k] for k in ["schema_version", *OPTIONS_JSON_SCHEMA]}
    # convert enums to names
    options["fit_fix"] = [m.name for m in FitFix(options["fit_fix"])]
    options["fit_mode"] = FitMode(options["fit_mode"]).name
    if options["trace"] is not None:
        options["trace"] = TraceChoice(options["trace"]).name
    return json.dumps(options)
