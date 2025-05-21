# Obtener la lista de lenguajes habilitados
# https://cmake.org/cmake/help/latest/prop_gbl/ENABLED_LANGUAGES.html
# https://cmake.org/cmake/help/latest/command/get_property.html
get_property(lenguajes_habilitados GLOBAL PROPERTY "ENABLED_LANGUAGES")

# Comprobar la presencia de C++ y CUDA
# https://cmake.org/cmake/help/latest/command/list.html
list(FIND lenguajes_habilitados "CXX" cxx_habilitado)
list(FIND lenguajes_habilitados "CUDA" cuda_habilitado)

# Incluir los subdirectorios de CMake que usan C++
# si el lenguaje ha sido habilitado
# https://cmake.org/cmake/help/latest/command/if.html
if(NOT (cxx_habilitado EQUAL "-1"))
    add_subdirectory(cpp)
    add_subdirectory(mpi)
    add_subdirectory(omp)
    add_subdirectory(mpi_workstealing)
endif()

# Incluir los subdirectorios de CMake que usan C++ y CUDA
# si ambos lenguajes han sido habilitados
if((NOT (cuda_habilitado EQUAL "-1")) AND (NOT (cxx_habilitado EQUAL "-1")))
    add_subdirectory(cuda)
endif()

# Limpiar variables temporales
unset(cxx_habilitado)
unset(cuda_habilitado)
