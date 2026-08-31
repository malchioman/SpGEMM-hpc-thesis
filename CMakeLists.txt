cmake_minimum_required(VERSION 3.21)

project(spmm_hpc_thesis LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

add_executable(spmm_two_sided
    src/matrix_market.cpp
    src/two_sided_spmm.cpp
)
target_compile_features(spmm_two_sided PRIVATE cxx_std_17)
target_link_libraries(spmm_two_sided PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(spmm_two_sided PRIVATE /W4 /permissive-)
else()
    target_compile_options(spmm_two_sided PRIVATE -Wall -Wextra -Wpedantic)
endif()
