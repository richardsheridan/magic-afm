import pathlib
import subprocess

filename = pathlib.Path(__file__).parent / "_version.py"


def get():
    return subprocess.run(
        ["git", "describe", "--dirty", "--long", "--tags"],
        capture_output=True,
        check=True,
    ).stdout.decode()[:-1]


def read():
    import runpy

    try:
        return runpy.run_path(filename)["__version__"]
    except FileNotFoundError:
        return ""


def write(version):
    version_str = f"""\
__version__ = '{version}'
"""
    with filename.open("w", encoding="utf8") as f:
        f.write(version_str)


if __name__ == "__main__":
    __version__ = get()
    print(__version__)

    if not read() == __version__:
        write(__version__)
        print("Version updated")
