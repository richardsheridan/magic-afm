@echo off
call %USERPROFILE%\Miniconda3\Scripts\activate.bat %USERPROFILE%\Miniconda3\envs\magic-afm-build\
python make_version.py
pyinstaller -y magic_gui.spec