@echo off
rem Windows build driver for the HIP backend (TheRock SDK): loads the MSVC environment, then configures and builds.
setlocal EnableDelayedExpansion
rem Locate vcvars64.bat through the VS installer instead of a machine-specific install path. VCVARS64 overrides it.
set "VCVARS="
if defined VCVARS64 set "VCVARS=%VCVARS64%"
if not defined VCVARS if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
  for /f "usebackq delims=" %%V in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath 2^>nul`) do set "VCVARS=%%V\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS (
  echo [ERROR] Cannot locate Visual Studio's vcvars64.bat. Set VCVARS64 to its full path.
  exit /b 1
)
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

rem Ask the installed TheRock SDK where its devel root is instead of hardcoding a machine-specific path.
rem ROCM_PATH still works as an override (also for a system ROCm installation).
set "ROCM_SDK="
for /f "usebackq delims=" %%R in (`rocm-sdk path --root 2^>nul`) do if not defined ROCM_SDK set "ROCM_SDK=%%R"
if not defined ROCM_SDK if defined ROCM_PATH set "ROCM_SDK=%ROCM_PATH%"
if not defined ROCM_SDK (
  echo [ERROR] Cannot locate the ROCm SDK. Install it with rocm-sdk, or set ROCM_PATH to its root.
  exit /b 1
)
if not exist "%ROCM_SDK%\include\hip\hip_runtime.h" (
  echo [ERROR] ROCm SDK at "%ROCM_SDK%" has no include\hip\hip_runtime.h
  exit /b 1
)
rem HIP_TARGET defaults to both supported architectures in one exe: gfx1103 (RDNA3, WMMA) and gfx1035
rem (RDNA2, dot-product leaves). Override it to build for a single architecture only.
if not defined HIP_TARGET set "HIP_TARGET=gfx1103;gfx1035"
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

rem build-win/ holds all compile-time intermediates (CMake, Ninja, object files); it is git-ignored.
set "BUILD=%SRC%\build-win"
set "CFG=%BUILD%\CMakeCache.txt"
rem dist/ holds the final compiled artifacts; it is git-ignored.
set "DIST=%SRC%\dist"

rem Reconfigure when HIP_TARGET changes: an old cache would otherwise silently build only the previous architectures.
set "RECONFIG="
set "CACHED_TARGET="
if exist "%CFG%" (
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b "HIP_TARGET:" "%CFG%"`) do set "CACHED_TARGET=%%B"
  if not "!CACHED_TARGET!"=="%HIP_TARGET%" set "RECONFIG=1"
) else (
  set "RECONFIG=1"
)

if defined RECONFIG (
  cmake -G Ninja -S "%SRC%" -B "%BUILD%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER="%ROCM_SDK%\lib\llvm\bin\clang-cl.exe" ^
    -DHIP_TARGET="%HIP_TARGET%" ^
    -DROCM_PATH="%ROCM_SDK%" ^
    -DMI_HIP_WERROR=OFF ^
    -DMOSSINFER_HIP_TOOLS=OFF
  if errorlevel 1 exit /b 1
)

cmake --build "%BUILD%" -j
if errorlevel 1 exit /b 1

if not exist "%DIST%" mkdir "%DIST%"
copy /y "%BUILD%\moss-infer-hip.exe" "%DIST%\" >nul
rem The SDK's versioned amdhip64*.dll needs these two DLLs next to it (rocm_kpack, amd_comgr).
copy /y "%BUILD%\amdhip64*.dll" "%DIST%\" >nul
copy /y "%BUILD%\rocm_kpack.dll" "%DIST%\" >nul
copy /y "%BUILD%\amd_comgr.dll" "%DIST%\" >nul
exit /b 0
