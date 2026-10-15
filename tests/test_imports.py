"""Public modules import without the optional dependencies installed."""

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

import os
import re
import subprocess
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Runs in a fresh interpreter: argv[1] is a comma-separated list of blocked
# top-level import names, argv[2:] are the modules to import under the block.
BLOCKED_IMPORT_SCRIPT = """
import importlib
import sys

blocked = frozenset(filter(None, sys.argv[1].split(",")))
preloaded = sorted(m for m in sys.modules if m.partition(".")[0] in blocked)
if preloaded:
    raise SystemExit(f"blocked modules already imported: {preloaded}")


class BlockOptional:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in blocked:
            raise ModuleNotFoundError(
                f"No module named {name!r} (blocked optional dependency)",
                name=name,
            )
        return None


sys.meta_path.insert(0, BlockOptional())
for module in sys.argv[2:]:
    importlib.import_module(module)
"""


def _normalize(dist):
    return re.sub(r"[-_.]+", "-", dist).lower()


def _requirement_name(requirement):
    return _normalize(re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0])


def optional_import_names(keep=()):
    """Top-level import names of every optional dependency, except those in keep."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    base = {_requirement_name(r) for r in project["dependencies"]}
    optional = {
        _requirement_name(r)
        for reqs in project["optional-dependencies"].values()
        for r in reqs
    }
    optional -= base | {_normalize(k) for k in keep}
    names = {dist.replace("-", "_") for dist in optional}
    for module, owners in packages_distributions().items():
        owners = {_normalize(o) for o in owners}
        if owners & optional and not owners & base:
            names.add(module)
    return sorted(names)


def import_with_blocked(blocked, *modules):
    env = os.environ.copy()
    env.pop("MAFM_MAX_CPUS", None)
    return subprocess.run(
        [sys.executable, "-c", BLOCKED_IMPORT_SCRIPT, ",".join(blocked), *modules],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_blocker_hides_installed_optional_dependencies():
    blocked = optional_import_names()
    assert {"psutil", "numba", "h5py", "trio", "trio_parallel"} <= set(blocked)
    with_trio = optional_import_names(keep=("trio",))
    assert "trio" not in with_trio
    assert "trio_parallel" in with_trio

    unblocked = import_with_blocked([], "psutil")
    assert unblocked.returncode == 0, unblocked.stderr
    for name in ["psutil", "trio", "trio_parallel", "h5py"]:
        proc = import_with_blocked(blocked, name)
        assert proc.returncode != 0, f"{name} imported despite the blocker"
        assert "ModuleNotFoundError" in proc.stderr, proc.stderr


@pytest.mark.parametrize(
    ("module", "keep"),
    [
        ("magic_afm.calculation", ()),
        ("magic_afm.data_readers", ()),
        ("magic_afm.async_tools", ("trio",)),
    ],
)
def test_module_imports_without_optional_dependencies(module, keep):
    proc = import_with_blocked(optional_import_names(keep), module)
    assert proc.returncode == 0, proc.stderr
