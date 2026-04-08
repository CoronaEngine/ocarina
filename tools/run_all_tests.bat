@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..\..") do set "ROOT_DIR=%%~fI"

set "BUILD_DIR=cmake-build-debug"
set "CONFIG=Debug"
set "DO_BUILD=1"
set "CONTINUE_ON_FAIL=1"
set "NO_PAUSE=0"

set "TEST_TARGETS=test-parsetype test-imageio test-half test-accel-update test-function-corrector test-counted_buffer test-buffer-access-modes test-bytebuffer-access-modes test-texture-access-modes test-trait test-soa test-bindless-array test-matrix-inverse test-swizzle test-texture-readwrite test-launch-latency test-printer test-printer-bytebuffer-debug test-printer-bindless-buffer-debug test-atomic"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--build-dir" (
    if "%~2"=="" (
        echo [ERROR] --build-dir requires a path.
        exit /b 2
    )
    set "BUILD_DIR=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--config" (
    if "%~2"=="" (
        echo [ERROR] --config requires a value.
        exit /b 2
    )
    set "CONFIG=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--no-build" (
    set "DO_BUILD=0"
    shift
    goto parse_args
)
if /I "%~1"=="--stop-on-fail" (
    set "CONTINUE_ON_FAIL=0"
    shift
    goto parse_args
)
if /I "%~1"=="--no-pause" (
    set "NO_PAUSE=1"
    shift
    goto parse_args
)
if /I "%~1"=="--help" (
    goto usage
)

echo [ERROR] Unknown argument: %~1
echo.
goto usage

:args_done
pushd "%ROOT_DIR%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to enter repository root.
    set "EXIT_CODE=2"
    goto finish
)

set "BIN_DIR=%BUILD_DIR%\bin"
set "FAILED_TESTS="
set /a PASSED_COUNT=0
set /a FAILED_COUNT=0
set /a TOTAL_COUNT=0

echo ============================================
echo   Ocarina Test Runner
echo ============================================
echo Root Dir  : .
echo Build Dir : %BUILD_DIR%
echo Config    : %CONFIG%
echo Build     : %DO_BUILD%
echo.

if not exist "%BUILD_DIR%" (
    echo [ERROR] Build directory does not exist: %BUILD_DIR%
    set "EXIT_CODE=2"
    goto finish
)

if "%DO_BUILD%"=="1" (
    call :setup_msvc
    if errorlevel 1 (
        set "EXIT_CODE=%ERRORLEVEL%"
        goto finish
    )

    echo [BUILD] Building Ocarina test targets...
    cmake --build "%BUILD_DIR%" --config %CONFIG% --target %TEST_TARGETS%
    if errorlevel 1 (
        echo [ERROR] Build failed.
        set "EXIT_CODE=%ERRORLEVEL%"
        goto finish
    )
    echo.
)

for %%T in (%TEST_TARGETS%) do (
    set /a TOTAL_COUNT+=1
    call :run_test %%T
    if errorlevel 1 (
        set /a FAILED_COUNT+=1
        if defined FAILED_TESTS (
            set "FAILED_TESTS=!FAILED_TESTS! %%T"
        ) else (
            set "FAILED_TESTS=%%T"
        )
        if "%CONTINUE_ON_FAIL%"=="0" (
            set "EXIT_CODE=1"
            goto summary
        )
    ) else (
        set /a PASSED_COUNT+=1
    )
)

if %FAILED_COUNT% GTR 0 (
    set "EXIT_CODE=1"
) else (
    set "EXIT_CODE=0"
)

:summary
echo.
echo ============================================
echo   Ocarina Test Summary
echo ============================================
echo Total  : %TOTAL_COUNT%
echo Passed : %PASSED_COUNT%
echo Failed : %FAILED_COUNT%
if defined FAILED_TESTS echo Failed Tests: %FAILED_TESTS%
echo ============================================

:finish
if not defined EXIT_CODE set "EXIT_CODE=0"

popd >nul 2>&1

if "%EXIT_CODE%"=="0" (
    echo [DONE] All Ocarina tests passed.
) else (
    echo [FAIL] Ocarina test run failed with exit code %EXIT_CODE%.
)

if "%NO_PAUSE%"=="0" pause
exit /b %EXIT_CODE%

:setup_msvc
set "VCVARS_BAT="

if defined VSINSTALLDIR (
    set "VCVARS_BAT=%VSINSTALLDIR%VC\Auxiliary\Build\vcvarsall.bat"
)

if not defined VCVARS_BAT (
    set "VSWHERE_EXE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "%VSWHERE_EXE%" (
        for /f "usebackq delims=" %%I in (`"%VSWHERE_EXE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            set "VCVARS_BAT=%%I\VC\Auxiliary\Build\vcvarsall.bat"
        )
    )
)

if not defined VCVARS_BAT (
    echo [ERROR] Failed to locate Visual Studio C++ build tools.
    exit /b 2
)

if not exist "%VCVARS_BAT%" (
    echo [ERROR] Found Visual Studio, but vcvarsall.bat is missing.
    exit /b 2
)

call "%VCVARS_BAT%" x64 >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to initialize MSVC build environment.
    exit /b %ERRORLEVEL%
)
exit /b 0

:run_test
set "TEST_NAME=%~1"
set "TEST_EXE=%TEST_NAME%.exe"

echo [RUN ] %TEST_NAME%
if not exist "%BIN_DIR%" (
    echo [FAIL] Missing binary directory: %BIN_DIR%
    exit /b 1
)

pushd "%BIN_DIR%" >nul
if errorlevel 1 (
    echo [FAIL] Failed to enter binary directory: %BIN_DIR%
    exit /b 1
)

if not exist "%TEST_EXE%" (
    echo [FAIL] Missing executable: %BIN_DIR%\%TEST_EXE%
    popd >nul
    exit /b 1
)

"%TEST_EXE%"
if errorlevel 1 (
    popd >nul
    echo [FAIL] %TEST_NAME%
    echo.
    exit /b 1
)

popd >nul

echo [PASS] %TEST_NAME%
echo.
exit /b 0

:usage
echo Usage: run_all_tests.bat [--build-dir PATH] [--config NAME] [--no-build] [--stop-on-fail] [--no-pause]
echo.
echo Defaults:
echo   --build-dir  cmake-build-debug
echo   --config     Debug
echo   build        enabled
echo   continue     enabled
echo.
exit /b 0