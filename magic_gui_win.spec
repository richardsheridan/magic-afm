# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('imageio')
block_cipher = None


a = Analysis(['magic_gui.py'],
             binaries=[
                 ('./_samplerate_data/libsamplerate-64bit.dll',r'./samplerate/_samplerate_data'),
                 ('./_samplerate_data/libsamplerate-32bit.dll',r'./samplerate/_samplerate_data'),
             ],
             datas=[('ARDFtoHDF5.exe','.'),('README.rst','.')],
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='magic_gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='magic_gui')
