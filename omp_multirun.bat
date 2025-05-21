@echo off
setlocal

if "%~1"=="" (
    echo Uso: %~nx0 NUM_VECES
    exit /b 1
)

set NUM_VECES=%1
set NUM_THREADS=%2
set NUM_ESFERAS=%3

echo Generating scene with %NUM_ESFERAS% spheres...
py genRandomScene.py %NUM_ESFERAS%

set CMD=".\out\raytracing_omp\raytracing_omp.exe" %NUM_THREADS% %*

for /L %%I in (1,1,%NUM_VECES%) do (
    echo [%%I/%NUM_VECES%] Executing...
    %CMD%
)

echo Finished!
endlocal
