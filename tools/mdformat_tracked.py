"""Run mdformat on every Markdown file tracked by git.

Extra arguments (e.g. ``--check``) are passed through to mdformat. Exits 0
when no Markdown files are tracked.
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


def main():
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        check=True,
        capture_output=True,
    ).stdout
    paths = [p for p in out.decode().split("\0") if p]
    if not paths:
        print("mdformat: no tracked Markdown files")
        return 0
    cmd = [sys.executable, "-m", "mdformat", *sys.argv[1:], *paths]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
