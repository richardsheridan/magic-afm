@echo off
call C:\Users\richard\Miniconda3\Scripts\activate.bat C:\Users\richard\Miniconda3\envs\magic-afm-build\
python make_version.py
pyinstaller -y magic_gui.spec
del _version.py