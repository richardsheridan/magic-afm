"""The public modules declare __all__ and underscore every other name."""

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

import ast
import importlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "magic fitting workflow.ipynb"

PUBLIC = {
    "magic_afm.calculation": frozenset(
        {
            "RESAMPLE_NPTS",
            "PROPERTY_UNITS_DICT",
            "PROPERTY_DTYPE",
            "PARMS_UNITS_DICT",
            "PARMS_DTYPE",
            "FitMode",
            "FitFix",
            "gauss3x3",
            "median3x1",
            "median3x3",
            "fillnan",
            "flatten",
            "planefit",
            "offset",
            "curve",
            "MANIPULATIONS",
            "resample_wrapper",
            "laser_interference",
            "hydrodynamic_drag",
            "schwarz_red",
            "lj_force",
            "lj_gradient",
            "red_extend",
            "red_retract",
            "red_both",
            "force_curve",
            "delta_curve",
            "rapid_forcecurve_estimate",
            "fit_force_curve",
            "pull_off_delta",
            "true_surface_z",
            "max_load_quantities",
            "perturb_k",
            "fit_curve_properties",
            "prepare_force_curve",
            "warmup_jit_main",
        }
    ),
    "magic_afm.data_readers": frozenset(
        {
            "Image",
            "Volume",
            "FVFile",
            "FileFormatError",
            "open_h5",
            "mmap_path_read_only",
            "convert_ardf",
            "ARH5ForceMapVolume",
            "ARH5Image",
            "ARH5FFMVolume",
            "ARH5File",
            "ARDFImage",
            "ARDFVolume",
            "ARDFFile",
            "NanoscopeImage",
            "NanoscopeVolume",
            "NanoscopeFile",
            "NIDVolume",
            "NIDMapHeightImage",
            "NIDMapImage",
            "NIDFile",
            "SUFFIX_FVFILE_MAP",
            "Index",
            "ZDArrays",
        }
    ),
    "magic_afm.async_tools": frozenset(
        {
            "TOOLTIP_CANCEL",
            "LONGEST_IMPERCEPTIBLE_DELAY",
            "cpu_bound_limiter",
            "to_sync_runner_map_unordered",
            "spinner_task",
            "tooltip_task",
            "receive_drain_and_get_latest",
        }
    ),
}

# Names that are renamed or underscored, so must no longer be module attributes.
REMOVED = {
    "magic_afm.calculation": frozenset(
        {
            "fitfun",
            "calc_properties_imap",
            "process_force_curve",
            "EPS",
            "RT_EPS",
            "gkern",
            "secant",
            "brentq",
            "mylinspace",
            "interp_with_offset",
            "mygradient",
            "mygradient_uniform",
            "curve_fit",
            "schwarz_wrap",
            "bigpow",
            "lilpow",
            "powrat",
            "prefactor",
            "postfactor",
            "lj_limit_factor",
            "check_jit",
            "warmup_jit_worker",
        }
    ),
    "magic_afm.data_readers": frozenset(
        {
            "NANOMETER_UNIT_CONVERSION",
            "TOC_STRUCT",
            "SI_PREFIX_EXPONENTS",
            "si_prefix_exponent",
            "nano_factor",
            "nm_per_unit",
            "nan_curve",
            "ChanMap",
            "StepInfo",
            "decode_cstring",
            "SEARCH_CHUNK_BYTES",
            "find_in_buffer",
            "parse_ar_note",
            "ARDFHeader",
            "ARDFTableOfContents",
            "ARDFTextTableOfContents",
            "ARDFVolumeTableOfContents",
            "ARDFVchan",
            "ARDFXdef",
            "ARDFVset",
            "ARDFVdata",
            "ARDFFFMReader",
            "ARDFForceMapReader",
            "parse_nanoscope_header",
            "nanoscope_curve_scale",
            "FFVReader",
            "QNMDReader",
            "QNMZReader",
            "NANOSURF_VALUE_REGEX",
            "NANOSURF_LINE_POINTS_REGEX",
            "NANOSURF_PARAM_TYPES",
            "NANOSURF_RAW_FRAMES",
            "NANOSURF_RESULT_IMAGES",
            "parse_nanosurf_value",
            "parse_nanosurf_line_points",
            "parse_nanosurf_header",
            "NIDDataset",
            "nid_map_line_to_rc",
            "nid_map_rc_to_line",
            "nid_map_from_lines",
        }
    ),
    "magic_afm.async_tools": frozenset({"MAX_WORKERS"}),
}

MODULES = sorted(PUBLIC)


def _bound_names(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _top_level_statements(body: list[ast.stmt]):
    # descend into try/if blocks (import fallbacks) but not into defs or classes
    for node in body:
        yield node
        if isinstance(node, ast.Try):
            yield from _top_level_statements(node.body)
            for handler in node.handlers:
                yield from _top_level_statements(handler.body)
            yield from _top_level_statements(node.orelse)
            yield from _top_level_statements(node.finalbody)
        elif isinstance(node, ast.If):
            yield from _top_level_statements(node.body)
            yield from _top_level_statements(node.orelse)


def _defined_and_imported(module_name: str) -> tuple[set[str], set[str]]:
    module = importlib.import_module(module_name)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    defined: set[str] = set()
    imported: set[str] = set()
    for node in _top_level_statements(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.partition(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                defined |= _bound_names(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            defined |= _bound_names(node.target)
    return defined, imported


@pytest.mark.parametrize("module_name", MODULES)
def test_all_lists_exactly_the_public_names(module_name):
    module = importlib.import_module(module_name)
    names = list(module.__all__)
    assert len(names) == len(set(names))
    assert set(names) == PUBLIC[module_name]


@pytest.mark.parametrize("module_name", MODULES)
def test_every_name_in_all_exists(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []


@pytest.mark.parametrize("module_name", MODULES)
def test_every_other_top_level_name_is_underscored(module_name):
    module = importlib.import_module(module_name)
    public = set(module.__all__)
    defined, imported = _defined_and_imported(module_name)
    stray = sorted(
        name for name in defined - imported - public if not name.startswith("_")
    )
    assert stray == []


@pytest.mark.parametrize("module_name", MODULES)
def test_renamed_and_underscored_names_are_gone(module_name):
    module = importlib.import_module(module_name)
    present = sorted(name for name in REMOVED[module_name] if hasattr(module, name))
    assert present == []


def _notebook_imports() -> list[tuple[str, str]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # drop IPython magics and shell escapes, which are not Python
        source = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("magic_afm")
            ):
                found.update((node.module, alias.name) for alias in node.names)
    return sorted(found)


def test_notebook_imports_from_magic_afm_resolve():
    imports = _notebook_imports()
    assert imports
    unresolved = [
        (module_name, name)
        for module_name, name in imports
        if not hasattr(importlib.import_module(module_name), name)
        or name in REMOVED.get(module_name, ())
    ]
    assert unresolved == []
