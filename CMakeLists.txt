cmake_minimum_required(VERSION 3.21)

project(spmm_hpc_thesis LANGUAGES CXX)

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

add_library(spmm_support STATIC
    src/spmm_common.cpp
    src/spmm_exchange.cpp
)
target_compile_features(spmm_support PUBLIC cxx_std_17)
target_include_directories(spmm_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(spmm_support PUBLIC matrix_market MPI::MPI_CXX OpenMP::OpenMP_CXX)
enable_project_warnings(spmm_support)

function(add_spmm_executable target_name source_file)
    add_executable(${target_name}
        ${source_file}
    )
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_link_libraries(${target_name} PRIVATE spmm_support)
    enable_project_warnings(${target_name})
endfunction()

add_spmm_executable(spmm_two_sided src/two_sided_spmm.cpp)
add_spmm_executable(spmm_one_sided_get src/one_sided_get_spmm.cpp)
add_spmm_executable(spmm_one_sided_put src/one_sided_put_spmm.cpp)

add_executable(matrix_market_smoke
    tests/matrix_market_smoke.cpp
)
target_compile_features(matrix_market_smoke PRIVATE cxx_std_17)
target_link_libraries(matrix_market_smoke PRIVATE matrix_market)
enable_project_warnings(matrix_market_smoke)

enable_testing()
add_test(
    NAME matrix_market_smoke
    COMMAND matrix_market_smoke ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/tiny_symmetric.mtx
)
