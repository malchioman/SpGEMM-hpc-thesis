cmake_minimum_required(VERSION 3.21)

project(spgemm_hpc_thesis LANGUAGES CXX)
include(CTest)

set(SPGEMM_MPI_CHUNK_LIMIT 2147483647 CACHE STRING "Maximum CSR elements per MPI message (lower for testing)")
if(NOT SPGEMM_MPI_CHUNK_LIMIT MATCHES "^[0-9]+$" OR
   SPGEMM_MPI_CHUNK_LIMIT LESS 1 OR SPGEMM_MPI_CHUNK_LIMIT GREATER 2147483647)
    message(FATAL_ERROR "SPGEMM_MPI_CHUNK_LIMIT must be in [1, 2147483647]")
endif()
add_compile_definitions(SPGEMM_MPI_CHUNK_LIMIT=${SPGEMM_MPI_CHUNK_LIMIT})

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

# Conservative automatic invalidation on reader/reference source or toolchain changes.
set(reference_identity "partial-reference-v1;${CMAKE_SYSTEM_PROCESSOR};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_RELEASE};${CMAKE_CXX_FLAGS_DEBUG};${CMAKE_CXX_FLAGS_RELWITHDEBINFO};${CMAKE_CXX_FLAGS_MINSIZEREL};${OpenMP_CXX_FLAGS};${CMAKE_BUILD_TYPE}")
foreach(source IN ITEMS src/common/spgemm_common.cpp src/common/spgemm_common.hpp
        src/common/partial_reference.cpp src/common/partial_reference.hpp
        src/common/full_reference.cpp src/common/full_reference.hpp src/common/prepare_full_reference.cpp
        src/common/matrix_market.cpp src/common/matrix_market.hpp src/common/csr_matrix.hpp CMakeLists.txt)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_hash)
    string(APPEND reference_identity ";${source_hash}")
endforeach()
string(SHA256 SPGEMM_REFERENCE_BUILD_ID "${reference_identity}")
configure_file(src/common/reference_build.hpp.in generated/reference_build.hpp @ONLY)

function(enable_project_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_library(matrix_market STATIC
    src/common/matrix_market.cpp
)
target_compile_features(matrix_market PUBLIC cxx_std_17)
target_include_directories(matrix_market PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
enable_project_warnings(matrix_market)

add_library(spgemm_support STATIC
    src/common/spgemm_common.cpp
    src/common/partial_reference.cpp
    src/common/full_reference.cpp
    src/common/benchmark_metrics.cpp
)
target_compile_features(spgemm_support PUBLIC cxx_std_17)
target_include_directories(spgemm_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
target_include_directories(spgemm_support PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_definitions(spgemm_support PRIVATE SPGEMM_REFERENCE_CONFIGURATION="$<CONFIG>")
target_link_libraries(spgemm_support PUBLIC matrix_market MPI::MPI_CXX OpenMP::OpenMP_CXX OpenSSL::Crypto)
enable_project_warnings(spgemm_support)

add_executable(prepare_partial_reference src/common/prepare_partial_reference.cpp)
target_link_libraries(prepare_partial_reference PRIVATE spgemm_support)
enable_project_warnings(prepare_partial_reference)

add_executable(prepare_full_reference src/common/prepare_full_reference.cpp)
target_link_libraries(prepare_full_reference PRIVATE spgemm_support)
enable_project_warnings(prepare_full_reference)

add_library(spgemm_baseline_support STATIC
    src/baselines/spgemm_exchange.cpp
)
target_compile_features(spgemm_baseline_support PUBLIC cxx_std_17)
target_include_directories(spgemm_baseline_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/baselines)
target_link_libraries(spgemm_baseline_support PUBLIC spgemm_support)
enable_project_warnings(spgemm_baseline_support)

function(add_baseline_executable target_name source_file)
    add_executable(${target_name}
        ${source_file}
    )
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_link_libraries(${target_name} PRIVATE spgemm_baseline_support)
    enable_project_warnings(${target_name})
endfunction()

add_baseline_executable(spgemm_two_sided src/baselines/two_sided_spgemm.cpp)
add_baseline_executable(spgemm_one_sided_get src/baselines/one_sided_get_spgemm.cpp)
add_baseline_executable(spgemm_one_sided_put src/baselines/one_sided_put_spgemm.cpp)

add_library(trident_support STATIC
    src/trident/common/process_grid.cpp
    src/trident/common/matrix_blocks.cpp
    src/trident/common/execution.cpp
    src/trident/common/csr_transfer.cpp
    src/trident/common/rma_get.cpp
    src/trident/common/intra_node_two_sided.cpp
    src/trident/common/benchmark.cpp
)
target_include_directories(trident_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/common)
target_link_libraries(trident_support PUBLIC spgemm_support)
enable_project_warnings(trident_support)

add_library(trident_two_sided_support STATIC src/trident/two_sided/inter_node_exchange.cpp)
target_include_directories(trident_two_sided_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/two_sided)
target_link_libraries(trident_two_sided_support PUBLIC trident_support)
enable_project_warnings(trident_two_sided_support)

add_executable(trident_two_sided src/trident/two_sided/main_two_sided.cpp)
target_link_libraries(trident_two_sided PRIVATE trident_two_sided_support)
enable_project_warnings(trident_two_sided)

add_library(trident_hybrid_support STATIC src/trident/hybrid/inter_node_exchange.cpp)
target_include_directories(trident_hybrid_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/hybrid)
target_link_libraries(trident_hybrid_support PUBLIC trident_support Threads::Threads)
enable_project_warnings(trident_hybrid_support)

add_executable(trident_hybrid src/trident/hybrid/main_hybrid.cpp)
target_link_libraries(trident_hybrid PRIVATE trident_hybrid_support)
enable_project_warnings(trident_hybrid)

add_library(trident_get_support STATIC src/trident/get/inter_node_exchange.cpp)
target_include_directories(trident_get_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/get)
target_link_libraries(trident_get_support PUBLIC trident_support)
enable_project_warnings(trident_get_support)

add_executable(trident_get src/trident/get/main_get.cpp)
target_link_libraries(trident_get PRIVATE trident_get_support)
enable_project_warnings(trident_get)

add_library(trident_get_pipeline_support STATIC src/trident/get_pipeline/inter_node_exchange.cpp)
target_include_directories(trident_get_pipeline_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/get_pipeline)
target_link_libraries(trident_get_pipeline_support PUBLIC trident_support)
enable_project_warnings(trident_get_pipeline_support)

add_executable(trident_get_pipeline src/trident/get_pipeline/main_get_pipeline.cpp)
target_link_libraries(trident_get_pipeline PRIVATE trident_get_pipeline_support)
enable_project_warnings(trident_get_pipeline)

add_library(trident_put_support STATIC src/trident/put/inter_node_exchange.cpp)
target_include_directories(trident_put_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/put)
target_link_libraries(trident_put_support PUBLIC trident_support)
enable_project_warnings(trident_put_support)

add_executable(trident_put src/trident/put/main_put.cpp)
target_link_libraries(trident_put PRIVATE trident_put_support)
enable_project_warnings(trident_put)

if(BUILD_TESTING)
    include(tests/tests.cmake)
endif()
