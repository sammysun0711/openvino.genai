@echo off
@setlocal
SETLOCAL EnableDelayedExpansion

:: command line arguments parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="--config" (
        set BUILD_TYPE=%~2
        shift
    ) else if "%1"=="-h" (
        goto usage
    ) else (
        echo Unrecognized option specified "%1"
        goto usage
    )
    shift
    goto :input_arguments_loop
)

set "DEMO_DIR=%~dp0"
echo "DEMO_DIR: %DEMO_DIR%"
echo "Initialize OpenVINO Runtime"

set OPENVINO_BACKEND_BUILD_TYPE="%BUILD_TYPE%"
call "%DEMO_DIR%openvino_24.0_rc1\setupvars.bat"
cmake -B build
cmake --build build --config "%BUILD_TYPE%"

echo Done.
exit /b

:usage
echo Build test sample with OpenVINO backend API
echo.
echo Options:
echo   -h                        Print the help message
echo   --config BUILD_TYPE       Specify config for build type with coorespoinding openvino backend api library, avaliable option: "Release" or "Debug"
exit /b