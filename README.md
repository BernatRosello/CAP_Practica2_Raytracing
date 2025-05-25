# CAP_Practica2_Raytracing
Para buildear basta con ejecutar CMake en la raiz del repositorio.
Se ha desarrollado en windows, para poder compilar y ejecutarlo en windows es necesario tener [MSMPI](https://www.microsoft.com/en-us/download/details.aspx?id=105289) instalado.

Comandos para renderizar:
- MPI:
``` mpi_multirun.bat [num_veces] [num_procesos] ['r(ow)' o 'c(olumn)'] [num_spheres] [width] [height] [num_samples] [filename] ```
- OMP: 
``` omp_multirun.bat [num_veces] [num_threads] [num_spheres] [width] [height] [num_samples] [filename] ```
- OMP: 
``` omp_worksteal_multirun.bat [num_veces][num_threads][num_spheres][width][height][num_samples][filename] ```
