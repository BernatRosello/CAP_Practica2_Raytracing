@echo off
setlocal

if "%~1"=="" (
    echo Uso: %~nx0 NUM_VECES
    exit /b 1
)

set NUM_VECES=%1
set NUM_ESFERAS=%2

echo Generating scene with %NUM_ESFERAS% spheres...
py genRandomScene.py %NUM_ESFERAS%

set CMD=".\out\raytracing_cuda\raytracing_cuda.exe" %*

for /L %%I in (1,1,%NUM_VECES%) do (
    echo [%%I/%NUM_VECES%] Executing...
    %CMD%
)

echo Finished!
endlocal
