# Especificar versión mínima de CMake
# CMake 3.20 es la versión más reciente que viene incluida con
# Visual Studio 2019
cmake_minimum_required(VERSION "3.20")

# Declaramos el proyecto de CMake
# En caso de que no haga falta alguno de los lenguajes listados,
# se puede deshabilitar simplemente eliminándolo de la lista
# https://cmake.org/cmake/help/latest/command/project.html
project(plantilla_cap LANGUAGES "CXX" "CUDA")

#################################################
# Opciones relacionadas con el soporte de CUDA
#################################################

#[[
    Especificar lista de arquitecturas CUDA a las que compilar para todos los objetivos.

    Esto sirve para especificarle al compilador de CUDA las versiones de CUDA para las cuales
    debe compilar el código en lugar de utilizar los valores por defecto. Sin embargo, antes
    que especificarlo directamente en la descripción del proyecto, es preferible hacerlo
    mediante la variable de entorno CUDAARCHS o dándole valor a la variable CMAKE_CUDA_ARCHITECTURES
    desde la terminal al llamar a CMake para configurar el proyecto.

    Esta opción puede ser utilizada para evitar que el compilador escoja una arquitectura
    no compatible con la GPU disponible en el equipo, lo cual provoca que todo código destinado
    a ser ejecutado en la GPU nunca se ejecute y que los programas actúen como si dicho código
    no existiera.

    NOTA: Si es necesario entregar un ejecutable compilado, es recomendable o bien
    preguntar que tarjeta gráfica se va a utilizar para ejecutarlo, consultar
    la versión de la Compute Capability que admite y especificarla o
    cambiar el valor a "all" (CMake >= 3.23) para que se compile el código para todas las
    arquitecturas posibles.

    https://cmake.org/cmake/help/latest/envvar/CUDAARCHS.html
    https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
    https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
    https://kaixih.github.io/nvcc-options/
    https://developer.nvidia.com/cuda-gpus
]]
# Si la variable CMAKE_CUDA_ARCHITECTURES aún no ha sido definida
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # No especificar ninguna arquitectura en concreto y permitir que el compilador decida
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

#[[
    Habilitar la compilación de código CUDA separable en todos los objetivos de compilación.
    Esto permite que el código escrito en un archivo CUDA sea compilado a archivos objeto
    diferentes y que estos puedan llamar a código CUDA definido externamente.

    https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_SEPARABLE_COMPILATION.html
    https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html
    https://developer.nvidia.com/blog/building-cuda-applications-cmake/
    https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/
    https://on-demand.gputechconf.com/gtc-express/2012/presentations/gpu-object-linking.pdf
]]
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Habilitar la generación de símbolos de depuración para de los kernel CUDA
# cuando se compila con las configuraciones Debug y RelWithDebInfo
# https://cmake.org/cmake/help/latest/command/add_compile_options.html
# https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-G>")

# Incluir de forma expresa del directorio estándar de archivos de encabezado de CUDA
# Esto ayuda a evitar falsos errores de IntelliSense en Visual Studio
# https://cmake.org/cmake/help/latest/command/include_directories.html
# https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html
include_directories(SYSTEM "$<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>")

#################################################
# Importación de dependencias
#################################################

# Buscar soporte de OpenMP para C++ como obligatorio
# Si no hace falta usar OpenMP, esta línea se puede comentar
# https://cmake.org/cmake/help/latest/command/find_package.html
# https://cmake.org/cmake/help/latest/module/FindOpenMP.html
find_package("OpenMP" REQUIRED COMPONENTS "CXX")

# Buscar una implementación de la Message Parsing Interface como obligatoria
# Si no hace falta usar MPI, esta línea se puede comentar
# https://cmake.org/cmake/help/latest/command/find_package.html
# https://cmake.org/cmake/help/latest/module/FindMPI.html
find_package("MPI" REQUIRED COMPONENTS "CXX")

# Buscar librerías del CUDA Toolkit como obligatorias
# https://cmake.org/cmake/help/latest/command/find_package.html
# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package("CUDAToolkit" REQUIRED)

#################################################
# Descripción del ejecutable a crear
#################################################

# Relegado a CMakeLists.txt internos a sub-carpetas
# Generamos un ejecutable por version.

#################################################
#################################################

# Incluir subdirectorio con los ejemplos
# Este comando llama al archivo CMakeLists.txt del directorio especificado
# https://cmake.org/cmake/help/latest/command/add_subdirectory.html
add_subdirectory(programs)
