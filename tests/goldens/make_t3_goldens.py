"""Write tests/goldens/t3_goldens.json from the current CLI (plan item T3).

Run from the repository root:

    pixi run -e test python tests/goldens/make_t3_goldens.py

pytest never runs this. Regenerating changes the baseline, so do it on purpose
and give the reason in the commit. The CLI runs, the map loading and the
summaries are imported from tests/test_cli.py, so the test and this script
compute alike. The stack the values were captured on is recorded under
"stack" in the JSON.

To measure spread between environments:

    ... make_t3_goldens.py --out OTHER.json
    ... make_t3_goldens.py --compare A.json B.json

--compare prints, per stored field, the largest absolute and relative
difference and any change in a NaN pattern or in the sampled pixels.
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

# The CLI's workers are spawned, and spawn re-runs this file as __mp_main__
# in each of them, so everything but the definitions sits under main().

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def f32(x):
    """Shortest repr that reads back to the same float32; NaN as null."""
    import numpy as np

    x = np.float32(x)
    return None if np.isnan(x) else float(str(x))


def generate(t3, resolve, tmp):
    options, summaries, full = {}, {}, {}
    for name, (relpath, args) in t3.CLI_RUNS.items():
        if name not in t3.NAMING or t3.NAMING[name][3] != "npy":
            continue
        stem, mode, trace, _ = t3.NAMING[name]
        path = Path(resolve(relpath))
        out = Path(tmp) / name
        result = t3.invoke_cli(path, out, args)
        assert result.exit_code == 0, result.output
        options_path = out / t3.options_name(stem, mode=mode, trace=trace)
        options[name] = json.loads(options_path.read_text())
        maps = t3.run_maps(t3.CliRun(path, out, result), name)
        if name in t3.SUMMARY_RUNS:
            pixels = t3.sample_pixels(t3.nan_mask(maps[t3.DEFAULT_MAPS[0]]))
            summaries[name] = {
                "relpath": relpath,
                "args": list(args),
                "pixels": pixels,
                "maps": {m: summary_json(t3.summarize(maps[m], pixels)) for m in maps},
            }
        if name in t3.FULL_RUNS:
            full[name] = {
                m: [[f32(x) for x in row] for row in a] for m, a in maps.items()
            }
    return {"stack": stack(), "options": options, "summaries": summaries, "full": full}


def summary_json(summary):
    return {
        k: v if k == "nan_mask" else [f32(x) for x in v] for k, v in summary.items()
    }


def stack():
    import os
    import platform
    from importlib.metadata import PackageNotFoundError, version

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
        "click": ver("click"),
        "imageio": ver("imageio"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pixi_environment": os.environ.get("PIXI_ENVIRONMENT_NAME"),
    }


def dumps(obj, indent=0):
    """JSON with lists of numbers or mask rows on one line each."""
    pad = " " * indent
    inner = " " * (indent + 1)
    if isinstance(obj, dict):
        items = [
            f"{inner}{json.dumps(k)}: {dumps(v, indent + 1)}" for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list) and any(isinstance(x, list | dict | str) for x in obj):
        items = [inner + dumps(x, indent + 1) for x in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj)


def leaves(obj, path=()):
    """Yield (path, value) for every numeric list, mask and pixel list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("stack", "relpath", "args", "options"):
                yield from leaves(v, (*path, k))
    else:
        yield path, obj


def compare(a_path, b_path):
    import numpy as np

    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    print("A:", a["stack"])
    print("B:", b["stack"])
    print("options equal:", a["options"] == b["options"])
    print(f"{'field':60} {'max abs':>10} {'max rel':>10} NaN")
    b_leaves = dict(leaves(b))
    for path, x in leaves(a):
        y = b_leaves[path]
        if path[-1] in ("nan_mask", "pixels"):
            if x != y:
                print(f"{'.'.join(path):60} DIFFERENT")
            continue
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        nan_same = np.array_equal(np.isnan(x), np.isnan(y))
        ok = ~np.isnan(x) & ~np.isnan(y)
        diff = np.abs(x[ok] - y[ok])
        absdiff = diff.max(initial=0.0)
        scale = np.abs(x[ok])
        reldiff = (diff / np.where(scale == 0, 1, scale)).max(initial=0.0)
        flag = "same" if nan_same else "DIFFERENT"
        print(f"{'.'.join(path):60} {absdiff:10.3g} {reldiff:10.3g} {flag}")


def main():
    sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "tests")]
    import test_cli as t3
    from conftest import resolve_real_data_file

    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--out", type=Path, default=t3.GOLDENS_PATH)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    with tempfile.TemporaryDirectory() as tmp:
        goldens = generate(t3, resolve_real_data_file, tmp)
    args.out.write_text(dumps(goldens) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
