import subprocess
import pathlib


filename = pathlib.Path(__file__).parent / "_version.py"


def get():
    return subprocess.run(
        ["git", "describe", "--dirty", "--long", "--tags"], capture_output=True, check=True
    ).stdout.decode()[:-1]


def check(version):
    try:
        with filename.open("r") as f:
            oldversion = f.read()
    except FileNotFoundError:
        return False
    oldversion = oldversion.split("=")[-1].strip()
    return oldversion == version


def write(version):
    version = "__version__ = '" + version + "'\n"
    with filename.open("w") as f:
        f.write(version)


if __name__ == "__main__":
    v = get()
    if not check(v):
        write(v)
