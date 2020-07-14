import subprocess
import pathlib


def make():
    version = "__version__ = '"

    completed = subprocess.run(["git", "describe", "--dirty", "--long"], capture_output=True, check=True)
    version += completed.stdout.decode()[:-1]+"'\n"
    filename = pathlib.Path(__file__).parent/"_version.py"
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
    make()
