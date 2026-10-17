"""Run every pre-merge check and summarize the results.

Steps: ``pixi run lint``, then the test suite in the ``test`` and
``test-nonumba`` environments, then ``pixi run typecheck`` (last, because
pyrefly reads the ``test`` environment's interpreter, which the test step
installs). Every step runs even if an earlier one fails.
The full output goes to ``build/gate/gate-<timestamp>.log`` (copied to
``build/gate/last.log``); the console gets one line per step plus the
relevant excerpts. Extra arguments are passed to pytest. Exits nonzero if any
step failed.

When every file in ``tests/real_data_manifest.txt`` is present under the
real-data root (``$MAFM_TEST_DATA``, else ``example.data``) at its recorded
size, the test steps run with ``MAFM_REQUIRE_REAL_DATA=1``. A passing gate
that tested exactly the committed tree (no extra arguments, no environment
variable that changes test selection or numerics, a clean working tree with
no untracked files at the start and the end, no tracked file written during
the run, HEAD's tree unchanged, complete real data) appends a line
``<tree hash> <timestamp>`` to ``gate-passed-trees`` in the git common
directory and prints ``stamped tree <hash>``; otherwise it prints
``no stamp: <reason>``. A failing gate prints neither.
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

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "build" / "gate"
TAIL_LINES = 30
PYTEST_ARGS = ["-p", "no:cacheprovider", "-rfEs", "--tb=short", "--color=no"]
SUMMARY_RE = re.compile(r"^=+ .* in [\d.]+s.* =+$")
SHORT_SUMMARY_RE = re.compile(r"^=+ short test summary info =+$")
MANIFEST = ROOT / "tests" / "real_data_manifest.txt"
STAMP_FILE = "gate-passed-trees"
BLOCKED_PREFIXES = ("PYTEST_", "NUMBA_", "OPENBLAS_", "NPY_", "OMP_", "MKL_", "MAFM_")
BLOCKED_NAMES = {
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONWARNINGS",
    "PYTHONOPTIMIZE",
    "PYTHONDEVMODE",
}
ALLOWED_NAMES = {"MAFM_TEST_DATA", "MAFM_REQUIRE_REAL_DATA", "MAFM_GUI_TESTS"}


def steps(pixi, extra):
    run = [pixi, "run", "--locked"]
    yield "lint", [*run, "lint"], False
    for env in ("test", "test-nonumba"):
        yield f"pytest {env}", [*run, "-e", env, "test", *PYTEST_ARGS, *extra], True
    yield "typecheck", [*run, "typecheck"], False


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def blocking_env_vars():
    return sorted(
        name
        for name in os.environ
        if name not in ALLOWED_NAMES
        and (name.startswith(BLOCKED_PREFIXES) or name in BLOCKED_NAMES)
    )


def missing_real_data():
    """Manifest paths absent from the real-data root at their recorded size."""
    root = Path(os.environ.get("MAFM_TEST_DATA") or ROOT / "example.data")
    missing = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, original, stored, _, relpath = line.split(maxsplit=4)
        path = root / "test_cases" / relpath
        xz = path.with_name(path.name + ".xz")
        if not (
            (path.is_file() and path.stat().st_size == int(original))
            or (xz.is_file() and xz.stat().st_size == int(stored))
        ):
            missing.append(relpath)
    return missing


def tree_state():
    """HEAD's tree and the reasons the working tree differs from it."""
    tree = git("rev-parse", "HEAD^{tree}").strip()
    reasons = []
    status = git("status", "--porcelain", "-z", "--untracked-files=all").split("\0")
    untracked = [entry[3:] for entry in status if entry.startswith("??")]
    if any(entry and not entry.startswith("??") for entry in status):
        reasons.append("working tree or index differs from HEAD")
    if untracked:
        reasons.append(f"untracked files: {', '.join(untracked)}")
    merging = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"],
        cwd=ROOT,
        capture_output=True,
    )
    if merging.returncode == 0:
        reasons.append("merge in progress")
    return tree, reasons


def written_since(start, tracked):
    written = []
    for name in tracked:
        try:
            if (ROOT / name).stat().st_mtime > start:
                written.append(name)
        except FileNotFoundError:
            pass
    return written


def write_stamp(tree):
    common = Path(git("rev-parse", "--git-common-dir").strip())
    stamp = ROOT / common / STAMP_FILE
    with open(stamp, "a", encoding="utf-8") as f:
        f.write(f"{tree} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")


def run_step(cmd, log, extra_env=None):
    env = dict(os.environ, NO_COLOR="1", **(extra_env or {}))
    lines = []
    with subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        for line in proc.stdout:
            log.write(line)
            lines.append(line.rstrip("\n"))
    return proc.returncode, lines


def pytest_excerpt(lines):
    """The short test summary section and the final summary line."""
    final = next(
        (i for i in reversed(range(len(lines))) if SUMMARY_RE.match(lines[i])), None
    )
    if final is None:
        return None
    start = next(
        (i + 1 for i in range(final) if SHORT_SUMMARY_RE.match(lines[i])), final
    )
    return lines[start : final + 1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    pixi = os.environ.get("PIXI_EXE") or shutil.which("pixi")
    if pixi is None:
        print("gate: cannot find pixi", file=sys.stderr)
        return 2
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / time.strftime("gate-%Y%m%d-%H%M%S.log")
    extra = sys.argv[1:]
    no_stamp = []
    if extra:
        no_stamp.append(f"extra pytest arguments: {subprocess.list2cmdline(extra)}")
    blocking = blocking_env_vars()
    if blocking:
        no_stamp.append(f"environment variables set: {', '.join(blocking)}")
    missing = missing_real_data()
    if missing:
        no_stamp.append(f"real data incomplete: {', '.join(missing)}")
    test_env = {} if missing else {"MAFM_REQUIRE_REAL_DATA": "1"}
    try:
        start_tree, reasons = tree_state()
        tracked = git("ls-files", "-z").split("\0")
    except (OSError, subprocess.CalledProcessError) as e:
        start_tree, reasons, tracked = None, [f"cannot read git state: {e}"], []
    no_stamp += [f"at start, {r}" for r in reasons]
    start = time.time()
    failed = []
    with open(log_path, "w", encoding="utf-8") as log:
        for name, cmd, is_pytest in steps(pixi, extra):
            log.write(f"===== gate step: {name}\n$ {subprocess.list2cmdline(cmd)}\n")
            log.flush()
            t0 = time.monotonic()
            code, lines = run_step(cmd, log, test_env if is_pytest else None)
            dt = time.monotonic() - t0
            ok = code == 0
            log.write(f"===== {name}: exit {code} in {dt:.1f}s\n\n")
            log.flush()
            print(f"{name}: {'PASS' if ok else 'FAIL'} ({dt:.1f}s)", flush=True)
            excerpt = pytest_excerpt(lines) if is_pytest else None
            if excerpt is None and not ok:
                excerpt = lines[-TAIL_LINES:]
            for line in excerpt or ():
                print(f"    {line}")
            if not ok:
                failed.append(name)
    shutil.copyfile(log_path, LOG_DIR / "last.log")
    print(f"full log: {log_path.relative_to(ROOT)} (copy: build/gate/last.log)")
    if failed:
        print(f"GATE FAILED: {', '.join(failed)}")
        return 1
    print("GATE PASSED")
    if start_tree is not None:
        try:
            end_tree, reasons = tree_state()
        except (OSError, subprocess.CalledProcessError) as e:
            end_tree, reasons = None, [f"cannot read git state: {e}"]
        no_stamp += [f"at end, {r}" for r in reasons]
        if end_tree != start_tree:
            no_stamp.append("HEAD's tree changed during the run")
        written = written_since(start, [t for t in tracked if t])
        if written:
            no_stamp.append(
                f"tracked files written during the run: {', '.join(written)}"
            )
    if not no_stamp:
        try:
            write_stamp(start_tree)
        except (OSError, subprocess.CalledProcessError) as e:
            no_stamp.append(f"cannot write the stamp: {e}")
    if no_stamp:
        print(f"no stamp: {'; '.join(no_stamp)}")
    else:
        print(f"stamped tree {start_tree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
