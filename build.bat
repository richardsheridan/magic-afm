call %USERPROFILE%\Miniconda3\Scripts\activate.bat || goto :EOF
call conda create -n mafm-build-tmp -y python=3.10
call conda activate mafm-build-tmp || goto :error
pip install -U -r requirements.txt
REM pip install -U -r requirements-nightly-mpl.txt
python magic_afm\make_version.py
pyinstaller -y magic_gui_win.spec
call conda deactivate
call conda remove -n mafm-build-tmp -y --all
call copy ARDFtoHDF5.exe dist\magic_gui
goto :EOF

:error
REM https://stackoverflow.com/a/8965092/4504950
echo Failed with error %errorlevel%
exit /b %errorlevel%