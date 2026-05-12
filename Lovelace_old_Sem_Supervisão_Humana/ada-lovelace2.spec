# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(
    ['scripts\\teste.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('model\\base\\vicuna-7b-v1.5', 'model\\base\\vicuna-7b-v1.5'),
        ('model\\ada-lovelace-lora', 'model\\ada-lovelace-lora'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ada-lovelace2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
