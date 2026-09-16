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
    src/common/matrix_market.cpp
)
target_compile_features(matrix_market PUBLIC cxx_std_17)
target_include_directories(matrix_market PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
enable_project_warnings(matrix_market)

add_library(spgemm_support STATIC
    src/common/spgemm_common.cpp
)
target_compile_features(spgemm_support PUBLIC cxx_std_17)
target_include_directories(spgemm_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/common)
target_link_libraries(spgemm_support PUBLIC matrix_market MPI::MPI_CXX OpenMP::OpenMP_CXX)
enable_project_warnings(spgemm_support)

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
    src/trident/common/benchmark.cpp
)
target_include_directories(trident_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/common)
target_link_libraries(trident_support PUBLIC spgemm_support)
enable_project_warnings(trident_support)

add_library(trident_two_sided_support STATIC src/trident/two_sided/intra_node_exchange.cpp)
target_include_directories(trident_two_sided_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trident/two_sided)
target_link_libraries(trident_two_sided_support PUBLIC trident_support)
enable_project_warnings(trident_two_sided_support)

add_executable(trident_two_sided src/trident/two_sided/main_two_sided.cpp)
target_link_libraries(trident_two_sided PRIVATE trident_two_sided_support)
enable_project_warnings(trident_two_sided)

add_executable(matrix_market_smoke
    tests/matrix_market_smoke.cpp
)
target_compile_features(matrix_market_smoke PRIVATE cxx_std_17)
target_link_libraries(matrix_market_smoke PRIVATE matrix_market)
enable_project_warnings(matrix_market_smoke)

add_executable(matrix_market_shape_smoke
    tests/matrix_market_shape_smoke.cpp
)
target_compile_features(matrix_market_shape_smoke PRIVATE cxx_std_17)
target_link_libraries(matrix_market_shape_smoke PRIVATE matrix_market)
enable_project_warnings(matrix_market_shape_smoke)

add_executable(spgemm_serial_smoke
    tests/spgemm_serial_smoke.cpp
)
target_compile_features(spgemm_serial_smoke PRIVATE cxx_std_17)
target_link_libraries(spgemm_serial_smoke PRIVATE spgemm_support)
enable_project_warnings(spgemm_serial_smoke)

add_executable(spgemm_validation_smoke
    tests/spgemm_validation_smoke.cpp
)
target_compile_features(spgemm_validation_smoke PRIVATE cxx_std_17)
target_link_libraries(spgemm_validation_smoke PRIVATE spgemm_support)
enable_project_warnings(spgemm_validation_smoke)

enable_testing()
add_test(
    NAME matrix_market_smoke
    COMMAND matrix_market_smoke ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/tiny_symmetric.mtx
)
add_test(
    NAME matrix_market_rectangular
    COMMAND matrix_market_shape_smoke ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/tiny_rectangular.mtx
            rectangular
)
foreach(symmetry IN ITEMS symmetric skew-symmetric hermitian)
    add_test(
        NAME matrix_market_reject_rectangular_${symmetry}
        COMMAND matrix_market_shape_smoke
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/invalid_rectangular_${symmetry}.mtx
                reject-nonsquare
    )
endforeach()
add_test(
    NAME spgemm_serial_smoke
    COMMAND spgemm_serial_smoke
)
add_test(
    NAME spgemm_validation_smoke
    COMMAND spgemm_validation_smoke
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
            --results ${CMAKE_CURRENT_BINARY_DIR}/spgemm_distributed_smoke_v2.tsv
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
            --results ${CMAKE_CURRENT_BINARY_DIR}/spgemm_one_sided_get_smoke_v2.tsv
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
            --results ${CMAKE_CURRENT_BINARY_DIR}/spgemm_one_sided_put_smoke_v2.tsv
)

set_tests_properties(spgemm_distributed_smoke spgemm_one_sided_get_smoke
                     spgemm_one_sided_put_smoke PROPERTIES
                     FAIL_REGULAR_EXPRESSION "validation=FAIL" TIMEOUT 60)

foreach(target_name IN ITEMS spgemm_two_sided spgemm_one_sided_get spgemm_one_sided_put trident_two_sided)
    if(target_name STREQUAL "trident_two_sided")
        set(expected_protocol trident_staged_csr_v1)
    else()
        set(expected_protocol prepared_halo_v2)
    endif()
    foreach(case IN ITEMS rectangular empty uneven)
        if(case STREQUAL "rectangular")
            set(case_args --matrix-a ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rectangular_a.mtx
                          --matrix-b ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rectangular_b.mtx)
            set(case_ranks 4)
        elseif(case STREQUAL "empty")
            set(case_args --matrix ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/empty.mtx)
            set(case_ranks 2)
        else()
            set(case_args --rows 19 --cols 23 --b-cols 13 --nnz-per-row 3 --b-nnz-per-row 2)
            set(case_ranks 3)
        endif()
        add_test(NAME ${target_name}_${case}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${case_ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${target_name}> ${MPIEXEC_POSTFLAGS}
                    ${case_args} --threads 2 --schedule dynamic --chunk 1
                    --warmup 0 --repeats 3 --trials 2
                    --results ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_${case}_v2.tsv)
        set_tests_properties(${target_name}_${case} PROPERTIES
                             FAIL_REGULAR_EXPRESSION "validation=FAIL" TIMEOUT 60
                             PROCESSORS ${case_ranks})
    endforeach()
    add_test(NAME ${target_name}_validation_failure_v2
        COMMAND ${CMAKE_COMMAND}
                -DMPIEXEC=${MPIEXEC_EXECUTABLE} -DNUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                "-DMPI_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPI_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
                -DPROGRAM=$<TARGET_FILE:${target_name}>
                -DMATRIX=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/overflow.mtx
                -DRESULTS=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_failure_v2.tsv
                -DEXPECTED_PROTOCOL=${expected_protocol}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_validation_failure.cmake)
    set_tests_properties(${target_name}_validation_failure_v2 PROPERTIES TIMEOUT 45 PROCESSORS 2)

    add_test(
        NAME ${target_name}_validation_failure
        COMMAND ${CMAKE_COMMAND}
                "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
                "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
                "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
                "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
                "-DBENCHMARK=$<TARGET_FILE:${target_name}>"
                "-DINPUT_MATRIX=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/overflow_product.mtx"
                "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_validation_failure.tsv"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_validation_failure.cmake
    )
    set_tests_properties(${target_name}_validation_failure PROPERTIES TIMEOUT 30)
    foreach(input_case IN ITEMS rectangular overflow incompatible)
        add_test(
            NAME ${target_name}_no_validate_${input_case}
            COMMAND ${CMAKE_COMMAND}
                    "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
                    "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
                    "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
                    "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
                    "-DBENCHMARK=$<TARGET_FILE:${target_name}>"
                    "-DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
                    "-DINPUT_CASE=${input_case}"
                    "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_no_validate_${input_case}.tsv"
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_no_validation.cmake
        )
        set_tests_properties(${target_name}_no_validate_${input_case} PROPERTIES TIMEOUT 30)
    endforeach()
    foreach(input_case IN ITEMS reuse_a explicit_b synthetic)
        add_test(
            NAME ${target_name}_matrix_sources_${input_case}
            COMMAND ${CMAKE_COMMAND}
                    "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
                    "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
                    "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
                    "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
                    "-DBENCHMARK=$<TARGET_FILE:${target_name}>"
                    "-DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
                    "-DINPUT_CASE=${input_case}"
                    "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_sources_${input_case}.tsv"
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_matrix_sources.cmake
        )
        set_tests_properties(${target_name}_matrix_sources_${input_case} PROPERTIES TIMEOUT 30)
    endforeach()
endforeach()

include(tests/trident_tests.cmake)
