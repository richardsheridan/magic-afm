"""Tests for the gate's stamp of trees that passed, run against a fake pixi."""

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
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="the gate stamp needs git on PATH"
)

BLOCKED_PREFIXES = ("PYTEST_", "NUMBA_", "OPENBLAS_", "NPY_", "OMP_", "MKL_", "MAFM_")
BLOCKED_NAMES = {
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONWARNINGS",
    "PYTHONOPTIMIZE",
    "PYTHONDEVMODE",
}

# manifest path -> (original size, stored size, form committed in the data repository)
DATA = {
    "raw_only.bin": (100, 40, "raw"),
    "sub/packed.bin": (200, 70, "xz"),
}

# The fake reads its instructions from, and logs each call to, FAKE_PIXI_DIR.
FAKE_PIXI = r"""
import json, os, pathlib, subprocess, sys, time

here = pathlib.Path(os.environ["FAKE_PIXI_DIR"])
control = json.loads((here / "control.json").read_text())
args = sys.argv[1:]
step = args[args.index("-e") + 1] if "-e" in args else args[-1]
with open(here / "log.jsonl", "a") as f:
    f.write(json.dumps({"step": step, "argv": args, "env": dict(os.environ)}) + "\n")
for op, path in control.get("actions", {}).get(step, []):
    p = pathlib.Path(path)
    if op == "create":
        p.write_text("new\n")
    elif op == "delete":
        p.unlink()
    elif op == "rewrite_same":
        p.write_bytes(p.read_bytes())
        future = time.time() + 5
        os.utime(p, (future, future))
    elif op == "commit_delete":
        subprocess.run(["git", "rm", "-q", path], check=True)
        subprocess.run(["git", "commit", "-q", "--no-verify", "-m", "rm"], check=True)
if step in ("test", "test-nonumba"):
    print("============ 3 passed, 1 skipped in 0.10s ============")
else:
    print("fake step ok")
sys.exit(1 if step in control.get("fail", []) else 0)
"""


def _write_size(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _clean_env(gitconfig: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(BLOCKED_PREFIXES)
        and k not in BLOCKED_NAMES
        and not k.startswith("GIT_")
    }
    env["GIT_CONFIG_GLOBAL"] = str(gitconfig)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _git(cwd: Path, env: dict[str, str], *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    """A committed repository with tools/, its data repository, and a fake pixi.

    The data repository holds a manifest and the files it lists, and the main
    repository pins its commit at tests/magic-afm-test-data. Built once per
    module; each test works on a copy of the repository and data.
    """
    base = tmp_path_factory.mktemp("gate-template")
    gitconfig = base / "gitconfig"
    gitconfig.write_text(
        "[user]\n\tname = Gate Test\n\temail = gate@example.invalid\n"
        "[commit]\n\tgpgsign = false\n[init]\n\tdefaultBranch = master\n"
        "[core]\n\tautocrlf = false\n"
    )
    env = _clean_env(gitconfig)

    fake = base / "fake"
    fake.mkdir()
    script = fake / "fake_pixi.py"
    script.write_text(FAKE_PIXI)
    if os.name == "nt":
        (fake / "pixi.cmd").write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = fake / "pixi"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)

    data = base / "data"
    lines = ["# synthetic manifest for the gate tests"]
    for relpath, (original, stored, form) in DATA.items():
        lines.append(f"{'0' * 64}  {original}  {stored}  delta+lzma2  {relpath}")
        if form == "raw":
            _write_size(data / relpath, original)
        else:
            _write_size(data / (relpath + ".xz"), stored)
    (data / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    _git(data, env, "init", "-q")
    _git(data, env, "add", ".")
    _git(data, env, "commit", "-q", "--no-verify", "-m", "data")
    data_commit = _git(data, env, "rev-parse", "HEAD")

    repo = base / "repo"
    # an uninitialized submodule's empty directory
    (repo / "tests" / "magic-afm-test-data").mkdir(parents=True)
    shutil.copytree(
        REPO_ROOT / "tools",
        repo / "tools",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(REPO_ROOT / ".gitignore", repo / ".gitignore")
    (repo / "notes.txt").write_text("committed\n")
    _git(repo, env, "init", "-q")
    _git(repo, env, "add", ".")
    gitlink = f"160000,{data_commit},tests/magic-afm-test-data"
    _git(repo, env, "update-index", "--add", "--cacheinfo", gitlink)
    _git(repo, env, "commit", "-q", "--no-verify", "-m", "initial")
    # keep every tracked file clearly older than any gate run
    past = time.time() - 60
    for path in repo.rglob("*"):
        if path.is_file() and ".git" not in path.relative_to(repo).parts:
            os.utime(path, (past, past))
    return base


class Gate:
    def __init__(self, template: Path, tmp_path: Path):
        self.tmp = tmp_path
        self.repo = tmp_path / "repo"
        self.data_root = (tmp_path / "data").resolve()
        self.fake_dir = tmp_path / "fake"
        shutil.copytree(template / "repo", self.repo)
        shutil.copytree(template / "data", self.data_root)
        self.fake_dir.mkdir()
        self.env = _clean_env(template / "gitconfig")
        launcher = "pixi.cmd" if os.name == "nt" else "pixi"
        self.env["PIXI_EXE"] = str(template / "fake" / launcher)
        self.env["FAKE_PIXI_DIR"] = str(self.fake_dir)
        self.env["MAFM_TEST_DATA"] = str(self.data_root)
        self.control({})

    def git(self, *args, cwd=None) -> str:
        return _git(cwd or self.repo, self.env, *args)

    def control(self, control: dict) -> None:
        (self.fake_dir / "control.json").write_text(json.dumps(control))

    def tree(self, cwd=None) -> str:
        return self.git("rev-parse", "HEAD^{tree}", cwd=cwd)

    def run(self, *args, cwd=None, **env_changes) -> subprocess.CompletedProcess:
        env = dict(self.env)
        for key, value in env_changes.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        cwd = cwd or self.repo
        return subprocess.run(
            [sys.executable, str(cwd / "tools" / "gate.py"), *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def stamp_lines(self) -> list[str]:
        path = self.repo / ".git" / "gate-passed-trees"
        if not path.is_file():
            return []
        return [line for line in path.read_text().splitlines() if line.strip()]

    def calls(self) -> list[dict]:
        log = self.fake_dir / "log.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()]

    def pytest_calls(self) -> list[dict]:
        return [c for c in self.calls() if c["step"] in ("test", "test-nonumba")]


@pytest.fixture
def gate(template, tmp_path) -> Gate:
    return Gate(template, tmp_path)


def _lines(result) -> list[str]:
    return result.stdout.splitlines()


def _no_stamp_reason(result) -> str:
    reasons = [line for line in _lines(result) if line.startswith("no stamp: ")]
    assert reasons, result.stdout
    reason = " ".join(line.removeprefix("no stamp: ") for line in reasons).strip()
    assert reason, result.stdout
    return reason


def _assert_no_stamp(gate, result) -> str:
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GATE PASSED" in _lines(result)
    assert not any(line.startswith("stamped tree") for line in _lines(result))
    assert gate.stamp_lines() == []
    return _no_stamp_reason(result)


def test_clean_passing_gate_stamps_head_tree_and_requires_real_data(gate):
    result = gate.run()
    tree = gate.tree()
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"stamped tree {tree}" in _lines(result)
    assert not any(line.startswith("no stamp") for line in _lines(result))
    stamps = gate.stamp_lines()
    assert len(stamps) == 1
    fields = stamps[0].split()
    assert fields[0] == tree
    assert len(fields) >= 2  # hash, then a timestamp
    calls = gate.pytest_calls()
    assert [c["step"] for c in calls] == ["test", "test-nonumba"]
    for call in calls:
        assert call["env"].get("MAFM_REQUIRE_REAL_DATA") == "1", call["step"]


def test_allowed_variables_and_ignored_files_still_stamp_and_append(gate):
    tree = gate.tree()
    stamp_file = gate.repo / ".git" / "gate-passed-trees"
    stamp_file.write_text(f"{tree} earlier\n")
    (gate.repo / "build").mkdir()
    (gate.repo / "build" / "before.txt").write_text("ignored\n")
    gate.control({"actions": {"lint": [["create", "build/during.txt"]]}})
    result = gate.run(
        MAFM_REQUIRE_REAL_DATA="1",
        MAFM_GUI_TESTS="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    assert f"stamped tree {tree}" in _lines(result), result.stdout
    assert [line.split()[0] for line in gate.stamp_lines()] == [tree, tree]


def test_stamp_from_a_worktree_goes_to_the_shared_git_directory(gate):
    wt = gate.tmp / "wt"
    gate.git("worktree", "add", "-q", "-b", "side", str(wt))
    result = gate.run(cwd=wt)
    tree = gate.tree(cwd=wt)
    assert f"stamped tree {tree}" in _lines(result), result.stdout
    assert [line.split()[0] for line in gate.stamp_lines()] == [tree]
    worktree_gitdir = gate.repo / ".git" / "worktrees" / "wt"
    assert not (worktree_gitdir / "gate-passed-trees").exists()


def test_default_data_root_is_the_submodule_in_the_repository(gate):
    submodule = gate.repo / "tests" / "magic-afm-test-data"
    submodule.rmdir()
    # a rename, since git's read-only object files stop rmtree on Windows
    gate.data_root.rename(submodule)
    result = gate.run(MAFM_TEST_DATA=None)
    assert f"stamped tree {gate.tree()}" in _lines(result), result.stdout


def test_uninitialized_submodule_prevents_a_stamp_and_the_requirement(gate):
    reason = _assert_no_stamp(gate, gate.run(MAFM_TEST_DATA=None))
    assert "SHA256SUMS.txt" in reason
    for call in gate.pytest_calls():
        assert call["env"].get("MAFM_REQUIRE_REAL_DATA") != "1", call["step"]


def test_data_root_at_another_commit_prevents_a_stamp(gate):
    (gate.data_root / "later.bin").write_bytes(b"later")
    gate.git("add", "later.bin", cwd=gate.data_root)
    gate.git("commit", "-q", "--no-verify", "-m", "later", cwd=gate.data_root)
    reason = _assert_no_stamp(gate, gate.run())
    assert "pins" in reason


def test_data_root_that_is_not_a_git_checkout_prevents_a_stamp(gate):
    (gate.data_root / ".git").rename(gate.data_root / "was-git")
    reason = _assert_no_stamp(gate, gate.run())
    assert "not a git checkout" in reason


def test_failing_gate_writes_no_stamp_and_prints_no_stamp_line(gate):
    gate.control({"fail": ["test"]})
    result = gate.run()
    assert result.returncode == 1
    assert not any(
        line.startswith(("stamped tree", "no stamp")) for line in _lines(result)
    )
    assert gate.stamp_lines() == []


def test_extra_pytest_arguments_prevent_a_stamp(gate):
    reason = _assert_no_stamp(gate, gate.run("-k", "foo"))
    assert "argument" in reason.lower()


@pytest.mark.parametrize(
    "name, value",
    [
        ("PYTEST_ADDOPTS", "-x"),
        ("MAFM_RESAMPLE_NPTS", "300"),
        ("NUMBA_DISABLE_JIT", "1"),
        ("PYTHONPATH", "elsewhere"),
    ],
)
def test_environment_variable_that_changes_tests_prevents_a_stamp(gate, name, value):
    reason = _assert_no_stamp(gate, gate.run(**{name: value}))
    assert name in reason


def test_no_stamp_reason_names_every_blocking_variable(gate):
    blocking = {
        "PYTEST_ADDOPTS": "-x",
        "MAFM_RESAMPLE_NPTS": "300",
        "MAFM_MAX_CPUS": "1",
        "NUMBA_DISABLE_JIT": "1",
        "OPENBLAS_CORETYPE": "HASWELL",
        "NPY_DISABLE_CPU_FEATURES": "AVX512F",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": "elsewhere",
        "PYTHONWARNINGS": "ignore",
        "PYTHONOPTIMIZE": "1",
        "PYTHONDEVMODE": "1",
        "PYTHONSTARTUP": "startup.py",
    }
    reason = _assert_no_stamp(gate, gate.run(**blocking))
    assert [name for name in blocking if name not in reason] == []


@pytest.mark.parametrize(
    "damage", ["missing file", "raw at stored size", "xz at original size"]
)
def test_incomplete_real_data_prevents_a_stamp_and_the_requirement(gate, damage):
    raw = gate.data_root / "raw_only.bin"
    packed = gate.data_root / "sub" / "packed.bin.xz"
    if damage == "missing file":
        packed.unlink()
    elif damage == "raw at stored size":
        _write_size(raw, DATA["raw_only.bin"][1])
    else:
        _write_size(packed, DATA["sub/packed.bin"][0])
    reason = _assert_no_stamp(gate, gate.run())
    assert "data" in reason.lower()
    for call in gate.pytest_calls():
        assert call["env"].get("MAFM_REQUIRE_REAL_DATA") != "1", call["step"]


def test_modified_tracked_file_at_start_prevents_a_stamp(gate):
    (gate.repo / "notes.txt").write_text("edited\n")
    reason = _assert_no_stamp(gate, gate.run())
    assert "working tree" in reason.lower()


def test_staged_change_prevents_a_stamp(gate):
    (gate.repo / "staged.txt").write_text("staged\n")
    gate.git("add", "staged.txt")
    reason = _assert_no_stamp(gate, gate.run())
    assert "working tree" in reason.lower()


def test_untracked_file_present_only_at_the_start_prevents_a_stamp(gate):
    (gate.repo / "stray.txt").write_text("stray\n")
    gate.control({"actions": {"lint": [["delete", "stray.txt"]]}})
    reason = _assert_no_stamp(gate, gate.run())
    assert "untracked" in reason.lower()


def test_untracked_file_created_during_the_run_prevents_a_stamp(gate):
    gate.control({"actions": {"typecheck": [["create", "stray.txt"]]}})
    reason = _assert_no_stamp(gate, gate.run())
    assert "untracked" in reason.lower()


def test_tracked_file_rewritten_during_the_run_prevents_a_stamp(gate):
    gate.control({"actions": {"lint": [["rewrite_same", "notes.txt"]]}})
    reason = _assert_no_stamp(gate, gate.run())
    assert "notes.txt" in reason


def test_head_tree_moving_during_the_run_prevents_a_stamp(gate):
    gate.control({"actions": {"lint": [["commit_delete", "notes.txt"]]}})
    _assert_no_stamp(gate, gate.run())


def test_stamp_write_error_is_a_no_stamp_reason_not_a_failure(gate):
    (gate.repo / ".git" / "gate-passed-trees").mkdir()
    result = gate.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(line.startswith("stamped tree") for line in _lines(result))
    _no_stamp_reason(result)
