"""Run the local (untracked) git hook rules for one stage.

Usage: ``local_hooks.py <stage> [args...]``

The rules live in ``$MAFM_LOCAL_HOOKS/<stage>.d/``, else in
``$(git rev-parse --git-common-dir)/local-hooks/<stage>.d/``, which worktrees share.
Each rule file there runs in name order with the remaining arguments: for the
``commit-msg`` stage that is the path of the commit message file, for ``pre-commit``
nothing (a rule asks ``git diff --cached --name-only`` for the staged files). Names
starting with ``.`` or ending with ``~`` are not rules. ``*.py`` rules run with this
interpreter; anything else runs directly, on POSIX only if it is executable and on
Windows only with a ``PATHEXT`` suffix. Exits 1 if any rule failed, 0 otherwise
(including when the stage has no rules).
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
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"local rules: cannot find the git common dir: {e}")
    return Path(common_dir) / "local-hooks"


def is_rule(path):
    """Whether ``path`` is a rule file this platform can run."""
    if path.name.startswith(".") or path.name.endswith("~") or not path.is_file():
        return False
    if path.suffix == ".py":
        return True
    if os.name == "nt":
        pathext = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
        return path.suffix.lower() in pathext.lower().split(";")
    return os.access(path, os.X_OK)


def main(argv):
    if not argv:
        sys.exit("usage: local_hooks.py <stage> [args...]")
    stage, *args = argv
    stage_dir = rules_root() / f"{stage}.d"
    if not stage_dir.is_dir():
        return 0
    failed = False
    for rule in sorted(stage_dir.iterdir()):
        if not is_rule(rule):
            continue
        if rule.suffix == ".py":
            cmd = [sys.executable, str(rule), *args]
        else:
            cmd = [str(rule), *args]
        try:
            returncode = subprocess.run(cmd).returncode
        except OSError as e:
            print(f"{rule.name}: {e}", file=sys.stderr)
            returncode = 1
        if returncode:
            print(f"local rule failed: {rule.name}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
