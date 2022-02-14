@echo off
call %USERPROFILE%\Miniconda3\Scripts\activate.bat
call conda create -n mafm-build-tmp -y python=3.9
call conda activate mafm-build-tmp
pip install -U -r requirements.txt
python magic_afm\make_version.py
pyinstaller -y magic_gui_win.spec
call conda deactivate
call conda remove -n mafm-build-tmp -y --all