"""Run every lint check, even after one fails, and fail if any failed.

Checks: ruff check (pyflakes, import sorting, the default pycodestyle errors),
ruff format, and mdformat on tracked Markdown.
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

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = [
    ("ruff check", ["-m", "ruff", "check"]),
    ("ruff format", ["-m", "ruff", "format", "--check"]),
    ("mdformat", [str(HERE / "mdformat_tracked.py"), "--check"]),
]


def main():
    failed = []
    for name, args in CHECKS:
        print(f"--- {name}", flush=True)
        if subprocess.run([sys.executable, *args]).returncode:
            failed.append(name)
    if failed:
        print(f"lint failed: {', '.join(failed)}")
        return 1
    print("lint passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
