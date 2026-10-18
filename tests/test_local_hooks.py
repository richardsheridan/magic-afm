"""tools/local_hooks.py runs the untracked local git hook rules for a stage."""

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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "local_hooks.py"


def run_hooks(cwd, *args, rules=None):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") and k != "MAFM_LOCAL_HOOKS"
    }
    if rules is not None:
        env["MAFM_LOCAL_HOOKS"] = str(rules)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def add_rule(rules, stage, name, body, mode=0o644):
    rule = rules / f"{stage}.d" / name
    rule.parent.mkdir(parents=True, exist_ok=True)
    rule.write_text(body, newline="\n")
    rule.chmod(mode)
    return rule


def log_rule(tag, exit_code=0):
    """A Python rule that appends ``tag`` to order.txt in the cwd."""
    return (
        f"with open('order.txt', 'a') as f:\n    f.write({tag!r} + '\\n')\n"
        f"raise SystemExit({exit_code})\n"
    )


def test_dotfiles_backups_and_dirs_are_not_rules(tmp_path):
    rules = tmp_path / "rules"
    add_rule(rules, "pre-commit", "10-ok.py", log_rule("10-ok"))
    add_rule(rules, "pre-commit", ".hidden.py", log_rule("hidden", exit_code=1))
    add_rule(rules, "pre-commit", "20-old.py~", log_rule("backup", exit_code=1))
    (rules / "pre-commit.d" / "30-dir.py").mkdir()
    result = run_hooks(tmp_path, "pre-commit", rules=rules)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "order.txt").read_text().split() == ["10-ok"]


def test_missing_stage_dir_exits_zero(tmp_path):
    rules = tmp_path / "rules"
    add_rule(rules, "pre-commit", "10-fail.py", "raise SystemExit(1)\n")
    result = run_hooks(tmp_path, "commit-msg", "msg", rules=rules)
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""


def test_failing_rule_fails_and_is_named(tmp_path):
    rules = tmp_path / "rules"
    add_rule(rules, "commit-msg", "10-ok.py", log_rule("10-ok"))
    add_rule(rules, "commit-msg", "20-bad.py", log_rule("20-bad", exit_code=3))
    add_rule(rules, "commit-msg", "30-after.py", log_rule("30-after"))
    result = run_hooks(tmp_path, "commit-msg", "msg", rules=rules)
    assert result.returncode == 1
    assert "local rule failed: 20-bad.py" in result.stderr
    assert "10-ok" not in result.stderr
    assert "30-after" not in result.stderr
    order = (tmp_path / "order.txt").read_text().split()
    assert order == ["10-ok", "20-bad", "30-after"]


def test_args_reach_rule(tmp_path):
    rules = tmp_path / "rules"
    body = "import json, sys\nopen('args.json', 'w').write(json.dumps(sys.argv[1:]))\n"
    add_rule(rules, "commit-msg", "10-args.py", body)
    result = run_hooks(tmp_path, "commit-msg", "a file", "-x", rules=rules)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "args.json").read_text()) == ["a file", "-x"]


def test_py_rule_runs_with_this_interpreter(tmp_path):
    rules = tmp_path / "rules"
    # not executable and no shebang: only the interpreter can run it
    body = "import sys\nopen('python.txt', 'w').write(sys.executable)\n"
    add_rule(rules, "pre-commit", "10-which.py", body)
    result = run_hooks(tmp_path, "pre-commit", rules=rules)
    assert result.returncode == 0, result.stderr
    assert os.path.samefile((tmp_path / "python.txt").read_text(), sys.executable)


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable bits")
def test_posix_runs_executables_and_skips_the_rest(tmp_path):
    rules = tmp_path / "rules"
    add_rule(
        rules, "commit-msg", "10-run", '#!/bin/sh\necho "$@" > ran.txt\n', mode=0o755
    )
    add_rule(rules, "commit-msg", "20-not-executable", "#!/bin/sh\nexit 1\n")
    (rules / "commit-msg.d" / "30-dir").mkdir()
    add_rule(rules, "commit-msg", "40-no-shebang", "not a program\n", mode=0o755)
    result = run_hooks(tmp_path, "commit-msg", "msg", rules=rules)
    assert result.returncode == 1
    assert (tmp_path / "ran.txt").read_text() == "msg\n"
    failures = [
        line for line in result.stderr.splitlines() if line.startswith("local rule")
    ]
    assert failures == ["local rule failed: 40-no-shebang"]


@pytest.mark.skipif(os.name != "nt", reason="Windows PATHEXT rules")
def test_windows_runs_pathext_files_and_skips_the_rest(tmp_path):
    rules = tmp_path / "rules"
    add_rule(rules, "commit-msg", "10-run.cmd", "@echo %1 > ran.txt\r\n")
    add_rule(rules, "commit-msg", "20-notes.txt", "not a program\n")
    add_rule(rules, "commit-msg", "30-script.sh", "#!/bin/sh\nexit 1\n")
    result = run_hooks(tmp_path, "commit-msg", "msg", rules=rules)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ran.txt").read_text().split() == ["msg"]


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git on PATH")


@needs_git
def test_default_rules_dir_is_in_git_common_dir(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    add_rule(repo / ".git" / "local-hooks", "pre-commit", "10-bad.py", "exit(1)\n")
    result = run_hooks(repo, "pre-commit")
    assert result.returncode == 1
    assert "local rule failed: 10-bad.py" in result.stderr


@needs_git
def test_linked_worktree_shares_the_rules(tmp_path):
    repo = tmp_path / "repo"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q", str(repo)], check=True)
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "root"], cwd=repo, check=True
    )
    worktree = tmp_path / "wt"
    subprocess.run([*git, "worktree", "add", "-q", str(worktree)], cwd=repo, check=True)
    add_rule(repo / ".git" / "local-hooks", "pre-commit", "10-bad.py", "exit(1)\n")
    result = run_hooks(worktree, "pre-commit")
    assert result.returncode == 1
    assert "local rule failed: 10-bad.py" in result.stderr


def test_outside_a_repo_fails_with_a_message(tmp_path):
    result = run_hooks(tmp_path, "pre-commit")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert result.stderr.startswith("local rules: cannot find the git common dir")
