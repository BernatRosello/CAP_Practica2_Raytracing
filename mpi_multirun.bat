@echo off
setlocal

if "%~1"=="" (
    echo Uso: %~nx0 NUM_VECES
    exit /b 1
)

set NUM_VECES=%1
set NUM_PROCESOS=%2
set RENDER_TYPE=%3
set NUM_ESFERAS=%4

shift
shift
shift
shift

echo Generating scene with %NUM_ESFERAS% spheres...
py genRandomScene.py %NUM_ESFERAS%

set CMD=mpiexec -n %NUM_PROCESOS% ".\out\raytracing_mpi\raytracing_mpi.exe" %RENDER_TYPE% %*

for /L %%I in (1,1,%NUM_VECES%) do (
    echo [%%I/%NUM_VECES%] Executing...
    %CMD%
)

echo Finished!
endlocal
