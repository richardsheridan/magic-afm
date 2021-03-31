import pathlib
import subprocess

filename = pathlib.Path(__file__).parent / "_version.py"


def get():
    return subprocess.run(
        ["git", "describe", "--dirty", "--long", "--tags"], capture_output=True, check=True
    ).stdout.decode()[:-1]


def read():
    try:
        with filename.open("r", encoding="utf8") as f:
            oldversion = f.read()
    except FileNotFoundError:
        return False
    return oldversion.split("=")[-1].strip()[1:-1]


def write(version):
    version = "__version__ = '" + version + "'\n"
    with filename.open("w", encoding="utf8") as f:
        f.write(version)


if __name__ == "__main__":
    v = get()
    if not read() == v:
        write(v)
