"""Write the fitted-value goldens of tests/test_goldens.py from the current code.

Run from the repository root:

    pixi run -e test python tests/goldens/make_t2_goldens.py

pytest never runs this. Regenerating changes the baseline, so do it on purpose
and give the reason in the commit. The builders and fit helpers are imported
from tests/test_goldens.py, so the test and this script compute alike. The
stack the values were captured on is recorded under "stack" in the JSON.

The synthetic values are shared by every platform and go to
<out-dir>/t2_goldens.json. Real-file (.nid) values depend on the platform and go
to <out-dir>/<pixi platform>/t2_goldens.json. --out-dir defaults to tests/goldens.
--platform-only writes the platform file alone and leaves the shared one as is.

Two more modes help measure spread between environments and stacks:

    ... make_t2_goldens.py --out-dir OTHER [--platform-only] [--warmup-jit]
    ... make_t2_goldens.py --compare A.json B.json

--warmup-jit calls calculation.warmup_jit_worker() first, the state of GUI
workers and CLI --jit, which the tests do not run. --compare prints, per
stored field, the largest absolute and relative difference and any change in
the NaN pattern. A and B may be shared or platform files.
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

import argparse
import json
import os
import platform
import sys
import types
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "tests")]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import test_goldens as t2  # noqa: E402
from conftest import GOLDENS_DIR, golden_platform, resolve_real_data_file  # noqa: E402

from magic_afm import calculation  # noqa: E402


def f32(x):
    """Shortest repr that reads back to the same float32; NaN as null."""
    x = np.float32(x)
    return None if np.isnan(x) else float(str(x))


def f64(x):
    x = float(x)
    return None if np.isnan(x) else x


def listify(a, conv):
    a = np.asarray(a)
    if a.ndim == 0:
        return conv(a)
    return [listify(x, conv) for x in a]


def fingerprint_json(fp):
    return {
        name: value if name == "split" else listify(value, f64)
        for name, value in fp.items()
    }


def fields_json(fields):
    return {
        group: {name: listify(value, f32) for name, value in values.items()}
        for group, values in fields.items()
    }


def stack():
    def ver(dist):
        try:
            return version(dist)
        except PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": ver("numpy"),
        "numba": ver("numba"),
        "llvmlite": ver("llvmlite"),
        "soxr": ver("soxr"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME"),
        "model_jitted": not isinstance(calculation.red_retract, types.FunctionType),
    }


def generate_shared():
    synthetic = {}
    for case in t2.SYNTHETIC_CASES:
        curve = t2.demo_curve(**t2.SYNTHETIC_CASES[case])
        entry = {"raw": fingerprint_json(t2.raw_fingerprint(case))}
        if case == "demo_r0_c0":
            entry["resampled"] = {
                mode.name: fingerprint_json(
                    t2.resampled_fingerprint(((0, 0), curve), mode)
                )
                for mode in t2.MODES
            }
        entry["fits"] = {
            mode.name: fields_json(t2.fit_synthetic(case, mode)) for mode in t2.MODES
        }
        synthetic[case] = entry
    return {"stack": stack(), "synthetic": synthetic}


def generate_platform():
    try:
        path = Path(resolve_real_data_file(t2.NID_RELPATH))
    except pytest.skip.Exception as e:
        print("no nid section:", e.msg)
        return None
    nid = {"relpath": t2.NID_RELPATH}
    for mode in t2.MODES:
        fp, maps = t2.nid_results(path, mode)
        nid[mode.name] = {
            "resampled": fingerprint_json(fp),
            "maps": fields_json(maps),
        }
    return {"stack": stack(), "nid": nid}


def dumps(obj, indent=0):
    """JSON with lists of numbers on one line, so a map row is one diff line."""
    pad = " " * indent
    inner = " " * (indent + 1)
    if isinstance(obj, dict):
        items = [
            f"{inner}{json.dumps(k)}: {dumps(v, indent + 1)}" for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list) and any(isinstance(x, list | dict) for x in obj):
        items = [inner + dumps(x, indent + 1) for x in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj)


def leaves(obj, path=()):
    """Yield (path, array) for every numeric list in a goldens tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("stack", "relpath", "split"):
                yield from leaves(v, (*path, k))
    elif isinstance(obj, list):
        yield path, np.array(obj, dtype=np.float64)
    elif isinstance(obj, float) or obj is None:
        yield path, np.array(obj, dtype=np.float64)


def compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    print("A:", a["stack"])
    print("B:", b["stack"])
    print(f"{'field':60} {'max abs':>10} {'max rel':>10} NaN")
    b_leaves = dict(leaves(b))
    for path, x in leaves(a):
        if path not in b_leaves:
            print(f"{'.'.join(path):60} {'missing in B':>21}")
            continue
        y = b_leaves[path]
        nan_same = np.array_equal(np.isnan(x), np.isnan(y))
        ok = ~np.isnan(x) & ~np.isnan(y)
        diff = np.abs(x[ok] - y[ok])
        absdiff = diff.max(initial=0.0)
        scale = np.abs(x[ok])
        reldiff = (diff / np.where(scale == 0, 1, scale)).max(initial=0.0)
        flag = "same" if nan_same else "DIFFERENT"
        print(f"{'.'.join(path):60} {absdiff:10.3g} {reldiff:10.3g} {flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=GOLDENS_DIR)
    parser.add_argument("--platform-only", action="store_true")
    parser.add_argument("--warmup-jit", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if args.warmup_jit:
        calculation.warmup_jit_worker()
    if not args.platform_only:
        write(args.out_dir / "t2_goldens.json", generate_shared())
    platform_part = generate_platform()
    if platform_part is not None:
        write(args.out_dir / golden_platform() / "t2_goldens.json", platform_part)


def write(path, goldens):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(goldens) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
