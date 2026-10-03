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

from .calculation import FitFix, FitMode


class TraceChoice(enum.IntEnum):
    RETRACE = 0
    TRACE = 1
    # BOTH = 2
    # ALL = -1


def trace_choice(v):
    # name (current) or legacy int (GUI-written)
    return TraceChoice[v] if isinstance(v, str) else TraceChoice(v)


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
    fit_fix=FitFix,
    fit_mode=FitMode.__getitem__,
)

NULLABLE_FIELDS = {"k", "defl_sens", "sync_dist", "trace"}


def load_options(fp):
    """Read an options JSON from an open file into a dict of validated values.

    Any subset of the keys in OPTIONS_JSON_SCHEMA may be present."""
    options = json.load(fp)
    for k, value in list(options.items()):
        if k in NULLABLE_FIELDS and value is None:
            continue
        try:
            validator = OPTIONS_JSON_SCHEMA[k]
        except KeyError:
            raise ValueError(f"Unknown key '{k}' in options_json") from None
        options[k] = validator(value)
    return options


def dump_options(options):
    """Convert a dict of options to the text of an options JSON.

    The keys must be exactly those of OPTIONS_JSON_SCHEMA, ensuring
    the CLI will be able to read it back in."""
    mismatched = options.keys() ^ OPTIONS_JSON_SCHEMA.keys()
    if mismatched:
        raise ValueError(f"Options do not match OPTIONS_JSON_SCHEMA: {mismatched}")
    options = {k: options[k] for k in OPTIONS_JSON_SCHEMA}
    # convert enums to names
    options["fit_mode"] = FitMode(options["fit_mode"]).name
    if options["trace"] is not None:
        options["trace"] = TraceChoice(options["trace"]).name
    return json.dumps(options)
