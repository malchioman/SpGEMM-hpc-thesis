cmake_minimum_required(VERSION 3.21)

project(spgemm_hpc_thesis LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

function(enable_project_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(matrix_market STATIC
    src/matrix_market.cpp
)
target_compile_features(matrix_market PUBLIC cxx_std_17)
target_include_directories(matrix_market PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
enable_project_warnings(matrix_market)

add_library(spgemm_support STATIC
    src/spgemm_common.cpp
    src/spgemm_exchange.cpp
)
target_compile_features(spgemm_support PUBLIC cxx_std_17)
target_include_directories(spgemm_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(spgemm_support PUBLIC matrix_market MPI::MPI_CXX OpenMP::OpenMP_CXX)
enable_project_warnings(spgemm_support)

function(add_spgemm_executable target_name source_file)
    add_executable(${target_name}
        ${source_file}
    )
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_link_libraries(${target_name} PRIVATE spgemm_support)
    enable_project_warnings(${target_name})
endfunction()

add_spgemm_executable(spgemm_two_sided src/two_sided_spgemm.cpp)
add_spgemm_executable(spgemm_one_sided_get src/one_sided_get_spgemm.cpp)
add_spgemm_executable(spgemm_one_sided_put src/one_sided_put_spgemm.cpp)

add_executable(matrix_market_smoke
    tests/matrix_market_smoke.cpp
)
target_compile_features(matrix_market_smoke PRIVATE cxx_std_17)
target_link_libraries(matrix_market_smoke PRIVATE matrix_market)
enable_project_warnings(matrix_market_smoke)

add_executable(spgemm_serial_smoke
    tests/spgemm_serial_smoke.cpp
)
target_compile_features(spgemm_serial_smoke PRIVATE cxx_std_17)
target_link_libraries(spgemm_serial_smoke PRIVATE spgemm_support)
enable_project_warnings(spgemm_serial_smoke)

enable_testing()
add_test(
    NAME matrix_market_smoke
    COMMAND matrix_market_smoke ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/tiny_symmetric.mtx
)
add_test(
    NAME spgemm_serial_smoke
    COMMAND spgemm_serial_smoke
)
add_test(
    NAME spgemm_distributed_smoke
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:spgemm_two_sided>
            --rows 8
            --cols 8
            --b-cols 8
            --nnz-per-row 2
            --b-nnz-per-row 2
            --threads 1
            --warmup 1
            --repeats 2
            --trials 1
            --results ${CMAKE_CURRENT_BINARY_DIR}/spgemm_distributed_smoke.tsv
)
add_test(
    NAME spgemm_one_sided_get_smoke
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:spgemm_one_sided_get>
            --rows 8
            --cols 8
            --b-cols 8
            --nnz-per-row 2
            --b-nnz-per-row 2
            --threads 1
            --warmup 1
            --repeats 2
            --trials 1
            --results ${CMAKE_CURRENT_BINARY_DIR}/spgemm_one_sided_get_smoke.tsv
)
add_test(
    NAME spgemm_one_sided_put_smoke
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:spgemm_one_sided_put>
            --rows 8
            --cols 8
            --b-cols 8
            --nnz-per-row 2
            --b-nnz-per-row 2
            --threads 1
            --warmup 1
            --repeats 2
            --trials 1
            --results ${CMAKE_CURRENT_BINARY_DIR}/spgemm_one_sided_put_smoke.tsv
)
