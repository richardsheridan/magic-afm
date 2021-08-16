@echo off
call %USERPROFILE%\Miniconda3\Scripts\activate.bat %USERPROFILE%\Miniconda3\envs\mafm-build\
python magic_afm\make_version.py
pyinstaller -y magic_gui_win.spec