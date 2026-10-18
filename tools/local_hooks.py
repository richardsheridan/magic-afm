"""Run the local (untracked) git hook rules for one stage.

Usage: ``local_hooks.py <stage> [args...]``

The rules live in ``$MAFM_LOCAL_HOOKS/<stage>.d/``, else in
``$(git rev-parse --git-common-dir)/local-hooks/<stage>.d/``, which worktrees share.
Each file there runs in name order with the remaining arguments: ``*.py`` rules with
this interpreter, anything else directly (on POSIX only if it is executable). Exits 1
if any rule failed, 0 otherwise (including when the stage has no rules).
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
import subprocess
import sys
from pathlib import Path


def rules_root():
    if root := os.environ.get("MAFM_LOCAL_HOOKS"):
        return Path(root)
    common_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(common_dir) / "local-hooks"


def main(argv):
    if not argv:
        sys.exit("usage: local_hooks.py <stage> [args...]")
    stage, *args = argv
    stage_dir = rules_root() / f"{stage}.d"
    if not stage_dir.is_dir():
        return 0
    failed = False
    for rule in sorted(stage_dir.iterdir()):
        if not rule.is_file():
            continue
        if rule.suffix == ".py":
            cmd = [sys.executable, str(rule), *args]
        elif os.name == "nt" or os.access(rule, os.X_OK):
            cmd = [str(rule), *args]
        else:
            continue
        try:
            returncode = subprocess.run(cmd).returncode
        except OSError as e:
            print(e, file=sys.stderr)
            returncode = 1
        if returncode:
            print(f"local rule failed: {rule.name}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
