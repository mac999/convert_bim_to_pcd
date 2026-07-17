@echo off
rem ---------------------------------------------------------------------------
rem run.bat - IFC(BIM) -> textured FBX + RGB LAS/LAZ point cloud converter
rem
rem NOTE: keep this file pure ASCII. cmd.exe reads .bat with the console code
rem page (CP949 here), so UTF-8 comments would be mangled into commands.
rem
rem   run.bat                     convert every IFC in input\ with defaults
rem   run.bat --spacing 0.03      any convert_ifc_to_las.py option is passed through
rem   run.bat --no-fx --no-fbx    (see "Key Options" in README.md)
rem   run.bat viewer              web viewer only, no conversion (default port 5013)
rem   run.bat textures            pre-download textures only
rem
rem Python is auto-detected from the venv_lmm conda env. To use another one:
rem   set "PYTHON=C:\path\to\python.exe"  &&  run.bat
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
cd /d "%~dp0"

if not defined PYTHON set "PYTHON=C:\ProgramData\miniconda3\envs\venv_lmm\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

if /i "%~1"=="viewer"   goto :viewer
if /i "%~1"=="textures" goto :textures

rem No arguments -> default conversion. Otherwise pass everything through.
set "ARGS=%*"
if not defined ARGS set "ARGS=-i ./input -o ./output -c ./config.json"
"%PYTHON%" convert_ifc_to_las.py %ARGS%
exit /b %ERRORLEVEL%

rem --- subcommands: drop the first arg, forward the rest ----------------------
:viewer
call :collect %*
"%PYTHON%" webviewer.py -o ./output -c ./config.json %ARGS%
exit /b %ERRORLEVEL%

:textures
call :collect %*
"%PYTHON%" texture_manager.py -c ./config.json -t ./textures %ARGS%
exit /b %ERRORLEVEL%

rem Collect every argument except the first (the subcommand name) into ARGS.
:collect
set "ARGS="
shift
:collect_loop
if "%~1"=="" exit /b 0
set "ARGS=%ARGS% %1"
shift
goto :collect_loop
