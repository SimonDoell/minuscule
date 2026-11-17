cmake_minimum_required(VERSION 4.2.3)
project(minuscule C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(minuscule
  src/main.c
)

# find_package(OpenMP REQUIRED)
# target_link_libraries(minuscule PRIVATE OpenMP::OpenMP_C)

target_compile_options(minuscule PRIVATE
  -fopenmp -Wall -Werror
  -Wpedantic -Wno-unused
)

target_link_libraries(minuscule PRIVATE m)