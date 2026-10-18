"""tools/local_hooks.sh runs the untracked local git hook rules for a stage."""

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
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "local_hooks.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("sh") is None or shutil.which("git") is None,
    reason="local_hooks.sh needs sh and git on PATH",
)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", path.as_posix()], check=True)
    return path


def run_hooks(repo, *args, rules=None):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") and k != "MAFM_LOCAL_HOOKS"
    }
    if rules is not None:
        env["MAFM_LOCAL_HOOKS"] = rules.as_posix()
    return subprocess.run(
        ["sh", SCRIPT.as_posix(), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def add_rule(rules, stage, name, body):
    rule = rules / f"{stage}.d" / name
    rule.parent.mkdir(parents=True, exist_ok=True)
    rule.write_text(f"#!/bin/sh\n{body}\n", newline="\n")
    rule.chmod(0o755)


def test_missing_stage_dir_exits_zero(repo, tmp_path):
    rules = tmp_path / "rules"
    add_rule(rules, "pre-commit", "10-fail", "exit 1")
    result = run_hooks(repo, "commit-msg", "msg", rules=rules)
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""


def test_failing_rule_fails_and_is_named(repo, tmp_path):
    rules = tmp_path / "rules"
    add_rule(rules, "commit-msg", "10-ok", "echo 10-ok >> order.txt")
    add_rule(rules, "commit-msg", "20-bad", "echo 20-bad >> order.txt; exit 3")
    add_rule(rules, "commit-msg", "30-after", "echo 30-after >> order.txt")
    result = run_hooks(repo, "commit-msg", "msg", rules=rules)
    assert result.returncode == 1
    assert "local rule failed: 20-bad" in result.stderr
    assert "10-ok" not in result.stderr
    assert "30-after" not in result.stderr
    order = (repo / "order.txt").read_text().split()
    assert order == ["10-ok", "20-bad", "30-after"]


def test_args_reach_rule(repo, tmp_path):
    rules = tmp_path / "rules"
    add_rule(rules, "commit-msg", "10-args", "printf '%s\\n' \"$@\" > args.txt")
    result = run_hooks(repo, "commit-msg", "a file", "-x", rules=rules)
    assert result.returncode == 0, result.stderr
    assert (repo / "args.txt").read_text().splitlines() == ["a file", "-x"]


def test_default_rules_dir_is_in_git_common_dir(repo):
    add_rule(repo / ".git" / "local-hooks", "pre-commit", "10-bad", "exit 1")
    result = run_hooks(repo, "pre-commit")
    assert result.returncode == 1
    assert "local rule failed: 10-bad" in result.stderr
