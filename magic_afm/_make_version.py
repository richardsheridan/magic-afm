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
__pep440_version__ = '{version.split('-')[0][1:]}'
"""
    with filename.open("w", encoding="utf8") as f:
        f.write(version_str)


if __name__ == "__main__":
    __version__ = get()

    if not read() == __version__:
        write(__version__)
