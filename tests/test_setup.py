"""Installation and repository-configuration checks.

The run.sh tests execute the real script in a temporary copy of the project.
They create a real virtual environment, but swap in an empty requirements file
and a fake ``curl`` so that no network access is needed. They exercise the
setup and recovery logic only, not the real MediaPipe/OpenCV install.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gesture_controller import config

ROOT = Path(__file__).resolve().parent.parent
RUN_SH = ROOT / "run.sh"

on_apple_silicon_mac = pytest.mark.skipif(
    sys.platform != "darwin" or os.uname().machine != "arm64",
    reason="run.sh only supports macOS on Apple silicon",
)


def _shell_var(name: str) -> str:
    m = re.search(rf'^{name}="([^"]*)"$', RUN_SH.read_text(), re.M)
    assert m, f"{name} not found in run.sh"
    return m.group(1)


# ------------------------------------------------------------- pinned versions
def test_model_pin_is_identical_in_run_sh_and_config():
    assert _shell_var("MODEL_URL") == config.MODEL_URL
    assert _shell_var("MODEL_SHA256") == config.MODEL_SHA256
    assert "/latest/" not in config.MODEL_URL
    assert re.fullmatch(r"[0-9a-f]{64}", config.MODEL_SHA256)


def test_downloaded_model_matches_pinned_checksum():
    model = config.DEFAULT_MODEL_PATH
    if not model.is_file():
        pytest.skip("model not downloaded yet (run ./run.sh --setup-only)")
    assert hashlib.sha256(model.read_bytes()).hexdigest() == config.MODEL_SHA256


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, sep, version = line.partition("==")
            assert sep, f"{path.name}: '{line}' is not pinned with =="
            pins[name.strip().lower()] = version.strip()
    return pins


def test_requirements_are_exactly_pinned_and_consistent_with_constraints():
    direct = _pins(ROOT / "requirements.txt")
    locked = _pins(ROOT / "constraints.txt")
    assert {"mediapipe", "opencv-contrib-python", "numpy"} <= direct.keys()
    for name, version in direct.items():
        assert locked.get(name) == version, f"{name}: requirements {version} vs constraints {locked.get(name)}"
    _pins(ROOT / "requirements-dev.txt")


# ------------------------------------------------------------------ .gitignore
@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize("path, ignored", [
    (".env", True), (".env.local", True), ("prod.env", True), (".netrc", True),
    ("id_ed25519", True), ("server.pem", True), ("signing.key", True),
    ("credentials.json", True), ("client_secret.json", True), ("secrets.yaml", True),
    ("models/hand_landmarker.task", True), (".venv/bin/python", True),
    (".env.example", False), ("config.example.toml", False), ("secrets.sample.yaml", False),
    ("run.sh", False), ("requirements.txt", False), ("constraints.txt", False),
    ("gesture_controller/config.py", False), ("tests/test_setup.py", False),
])
def test_gitignore(path, ignored):
    res = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert (res.returncode == 0) is ignored


# ---------------------------------------------------------------------- run.sh
GOOD_MODEL = b"fake hand model\n"
FAKE_CURL = """#!/bin/sh
# Fake curl: write $FAKE_MODEL_SRC to the -o target, or fail if FAKE_CURL_FAIL=1.
[ "${FAKE_CURL_FAIL:-0}" = "1" ] && exit 22
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift; fi
  shift
done
cp "$FAKE_MODEL_SRC" "$out"
"""


class Project:
    """A throwaway copy of the launcher with a stub app and offline downloads."""

    def __init__(self, tmp: Path) -> None:
        self.dir = tmp / "proj"
        self.dir.mkdir()
        script = RUN_SH.read_text()
        sha = hashlib.sha256(GOOD_MODEL).hexdigest()
        script = re.sub(r'^MODEL_SHA256=".*"$', f'MODEL_SHA256="{sha}"', script, flags=re.M)
        (self.dir / "run.sh").write_text(script)
        (self.dir / "run.sh").chmod(0o755)
        self.set_requirements("# nothing to install in tests\n")
        (self.dir / "constraints.txt").write_text("")
        (self.dir / "main.py").write_text("import sys; print('APP STARTED', sys.argv[1:])\n")
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "curl").write_text(FAKE_CURL)
        (bin_dir / "curl").chmod(0o755)
        self.served = tmp / "served_model"
        self.serve(GOOD_MODEL)
        self.env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                    "FAKE_MODEL_SRC": str(self.served), "PYTHON": sys.executable}
        self.env.pop("VIRTUAL_ENV", None)

    def set_requirements(self, text: str) -> None:
        (self.dir / "requirements.txt").write_text(text)

    def serve(self, data: bytes) -> None:
        self.served.write_bytes(data)

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(["./run.sh", *args], cwd=self.dir, env={**self.env, **env},
                              capture_output=True, text=True, timeout=180)

    @property
    def model(self) -> Path:
        return self.dir / "models" / "hand_landmarker.task"

    @property
    def stamp(self) -> Path:
        return self.dir / ".venv" / ".requirements.sha256"


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path)


def _ok(res: subprocess.CompletedProcess) -> None:
    assert res.returncode == 0, res.stdout + res.stderr


@on_apple_silicon_mac
def test_fresh_setup_then_launch_passes_arguments(project):
    _ok(res := project.run("--setup-only"))
    assert "Setup complete." in res.stdout
    assert project.stamp.is_file()
    assert project.model.read_bytes() == GOOD_MODEL
    assert not project.model.with_suffix(".task.part").exists()

    _ok(res := project.run("--dry-run", "-v"))
    assert "Installing dependencies" not in res.stdout      # nothing to redo
    assert "Downloading" not in res.stdout
    assert "APP STARTED ['--dry-run', '-v']" in res.stdout


@on_apple_silicon_mac
def test_failed_dependency_install_is_retried_on_next_run(project):
    # Simulates the original bug: the venv exists after a failed install, and
    # the old script never tried to install again.
    project.set_requirements("./does-not-exist\n")
    res = project.run("--setup-only")
    assert res.returncode != 0
    assert "dependency installation failed" in res.stderr
    assert (project.dir / ".venv" / "bin" / "python").exists()
    assert not project.stamp.exists()

    project.set_requirements("# fixed\n")
    _ok(res := project.run("--setup-only"))
    assert "Installing dependencies" in res.stdout
    assert project.stamp.is_file()


@on_apple_silicon_mac
def test_changed_requirements_trigger_reinstall(project):
    _ok(project.run("--setup-only"))
    project.set_requirements("# edited\n")
    _ok(res := project.run("--setup-only"))
    assert "Installing dependencies" in res.stdout


@on_apple_silicon_mac
def test_broken_virtualenv_is_recreated(project):
    _ok(project.run("--setup-only"))
    vpy = project.dir / ".venv" / "bin" / "python"
    vpy.unlink()
    vpy.write_text("#!/bin/sh\nexit 1\n")   # e.g. its base Python was uninstalled
    vpy.chmod(0o755)
    _ok(res := project.run("--setup-only"))
    assert "recreating" in res.stdout
    assert "Installing dependencies" in res.stdout
    assert subprocess.run([str(vpy), "-c", "pass"]).returncode == 0


@on_apple_silicon_mac
def test_checksum_mismatch_is_rejected_and_not_kept(project):
    project.serve(b"tampered or truncated download")
    res = project.run("--setup-only")
    assert res.returncode != 0
    assert "expected" in res.stderr and "Not using it" in res.stderr
    assert not project.model.exists()
    assert not project.model.with_suffix(".task.part").exists()

    project.serve(GOOD_MODEL)
    _ok(project.run("--setup-only"))
    assert project.model.read_bytes() == GOOD_MODEL


@on_apple_silicon_mac
def test_failed_download_leaves_nothing_behind_and_is_retried(project):
    res = project.run("--setup-only", FAKE_CURL_FAIL="1")
    assert res.returncode != 0
    assert "model download failed" in res.stderr
    assert not project.model.exists()
    _ok(project.run("--setup-only"))
    assert project.model.read_bytes() == GOOD_MODEL


@on_apple_silicon_mac
def test_corrupt_existing_model_is_replaced(project):
    _ok(project.run("--setup-only"))
    project.model.write_bytes(GOOD_MODEL[:5])   # e.g. an interrupted download by the old script
    _ok(res := project.run("--setup-only"))
    assert "does not match the pinned checksum" in res.stdout
    assert project.model.read_bytes() == GOOD_MODEL


@on_apple_silicon_mac
def test_too_old_python_is_refused(project, tmp_path):
    old = tmp_path / "python3.9"
    old.write_text("#!/bin/sh\n"
                   "case \"$2\" in *platform*) echo 3.9.18;; *) exit 1;; esac\n")
    old.chmod(0o755)
    res = project.run("--setup-only", PYTHON=str(old))
    assert res.returncode != 0
    assert "Python 3.12 or newer is required" in res.stderr
    assert "3.9.18" in res.stderr
    assert not (project.dir / ".venv").exists()
