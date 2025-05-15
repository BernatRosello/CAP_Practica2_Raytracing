# Set the current directory as the root for this program version
set(PROGRAM_NAME "raytracing_cpp")

# Set source and include directories relative to this CMakeLists.txt
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Recursively gather source and header files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${SRC_DIR}/*.cpp"
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.hpp"
)

# Define the target (can be changed to add_library if needed)
add_executable(${PROGRAM_NAME} ${SOURCES})

# Specify include directories
target_include_directories(${PROGRAM_NAME} PRIVATE ${INCLUDE_DIR})

# Set output directory to your /out folder
set_target_properties(${PROGRAM_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out/${PROGRAM_NAME}"
)