import subprocess
import pathlib


def get():
    return subprocess.run(
        ["git", "describe", "--dirty", "--long", "--tags"], capture_output=True, check=True
    ).stdout.decode()[:-1]


def write():
    version = "__version__ = '"

    version += get() + "'\n"
    filename = pathlib.Path(__file__).parent / "_version.py"
    try:
        with filename.open("r") as f:
            oldversion = f.read()
        if oldversion == version:
            return
    except FileNotFoundError:
        pass
    with filename.open("w") as f:
        f.write(version)


if __name__ == "__main__":
    write()
