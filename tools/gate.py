"""Run every pre-merge check and summarize the results.

Steps: ``pixi run lint``, then the test suite in the ``test`` and
``test-nonumba`` environments, then ``pixi run typecheck`` (last, because
pyrefly reads the ``test`` environment's interpreter, which the test step
installs). Every step runs even if an earlier one fails.
The full output goes to ``build/gate/gate-<timestamp>.log`` (copied to
``build/gate/last.log``); the console gets one line per step plus the
relevant excerpts. Extra arguments are passed to pytest. Exits nonzero if any
step failed.
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


def steps(pixi, extra):
    run = [pixi, "run", "--frozen"]
    yield "lint", [*run, "lint"], False
    for env in ("test", "test-nonumba"):
        yield f"pytest {env}", [*run, "-e", env, "test", *PYTEST_ARGS, *extra], True
    yield "typecheck", [*run, "typecheck"], False


def run_step(cmd, log):
    env = dict(os.environ, NO_COLOR="1")
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
    failed = []
    with open(log_path, "w", encoding="utf-8") as log:
        for name, cmd, is_pytest in steps(pixi, sys.argv[1:]):
            log.write(f"===== gate step: {name}\n$ {subprocess.list2cmdline(cmd)}\n")
            log.flush()
            t0 = time.monotonic()
            code, lines = run_step(cmd, log)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
