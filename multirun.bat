@echo off
setlocal

if "%~1"=="" (
    echo Uso: %~nx0 NUM_VECES
    exit /b 1
)

set NUM_VECES=%1
set CMD=mpiexec -n 4 ".\out\raytracing_mpi\raytracing_mpi.exe"

for /L %%I in (1,1,%NUM_VECES%) do (
    echo [%%I/%NUM_VECES%] Executing...
    %CMD%
)

echo Finished!
pause
endlocal
