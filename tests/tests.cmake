add_executable(full_reference_test tests/full_reference_test.cpp)
target_link_libraries(full_reference_test PRIVATE spgemm_support)
enable_project_warnings(full_reference_test)
add_test(NAME full_reference COMMAND full_reference_test ${CMAKE_CURRENT_BINARY_DIR}/full-reference-test)

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

add_executable(spgemm_options_smoke tests/spgemm_options_smoke.cpp)
target_link_libraries(spgemm_options_smoke PRIVATE spgemm_support)
enable_project_warnings(spgemm_options_smoke)

add_executable(spgemm_full_streaming tests/spgemm_full_streaming.cpp)
target_link_libraries(spgemm_full_streaming PRIVATE spgemm_support)
enable_project_warnings(spgemm_full_streaming)
add_test(NAME spgemm_full_streaming COMMAND spgemm_full_streaming)
add_executable(partial_reference_test tests/partial_reference_test.cpp)
target_link_libraries(partial_reference_test PRIVATE spgemm_support)
enable_project_warnings(partial_reference_test)
add_test(NAME partial_reference COMMAND partial_reference_test ${CMAKE_CURRENT_BINARY_DIR}/reference-unit-cache)
add_executable(csr_large_counts tests/csr_large_counts.cpp)
target_link_libraries(csr_large_counts PRIVATE spgemm_support)
enable_project_warnings(csr_large_counts)
add_test(NAME csr_large_counts COMMAND csr_large_counts)
add_executable(csr_large_transfer tests/csr_large_transfer.cpp)
target_link_libraries(csr_large_transfer PRIVATE spgemm_support)
enable_project_warnings(csr_large_transfer)
add_test(NAME csr_large_transfer
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:csr_large_transfer> ${MPIEXEC_POSTFLAGS})
set_tests_properties(csr_large_transfer PROPERTIES TIMEOUT 60 PROCESSORS 2)
add_executable(matrix_market_contract tests/matrix_market_contract.cpp)
target_link_libraries(matrix_market_contract PRIVATE matrix_market)
enable_project_warnings(matrix_market_contract)
add_test(NAME matrix_market_contract COMMAND matrix_market_contract ${CMAKE_CURRENT_BINARY_DIR}/reader_contract.mtx)
add_test(NAME spgemm_options_smoke COMMAND spgemm_options_smoke)
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

foreach(target_name IN ITEMS spgemm_two_sided spgemm_one_sided_get spgemm_one_sided_put trident_two_sided trident_hybrid trident_get trident_get_pipeline trident_put)
    add_test(NAME ${target_name}_reject_malformed_numbers
        COMMAND ${CMAKE_COMMAND}
                "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
                "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
                "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
                "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
                "-DBENCHMARK=$<TARGET_FILE:${target_name}>"
                "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_invalid_numbers.tsv"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_numeric_options.cmake)
    set_tests_properties(${target_name}_reject_malformed_numbers PROPERTIES TIMEOUT 75 PROCESSORS 2)
    if(target_name STREQUAL "trident_two_sided")
        set(expected_protocol trident_staged_csr64_v2)
    elseif(target_name STREQUAL "trident_hybrid")
        set(expected_protocol trident_hybrid_rma_requests_csr64_v2)
    elseif(target_name STREQUAL "trident_get")
        set(expected_protocol trident_get_csr64_v2)
    elseif(target_name STREQUAL "trident_get_pipeline")
        set(expected_protocol trident_get_pipeline_csr64_v2)
    elseif(target_name STREQUAL "trident_put")
        set(expected_protocol trident_put_staged_csr64_v2)
    else()
        set(expected_protocol prepared_halo_csr64_v3)
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
    foreach(input_case IN ITEMS rectangular empty cancellation overflow)
        add_test(NAME ${target_name}_partial_${input_case}
            COMMAND ${CMAKE_COMMAND}
                    "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}"
                    "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
                    "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}"
                    "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
                    "-DBENCHMARK=$<TARGET_FILE:${target_name}>"
                    "-DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
                    "-DINPUT_CASE=${input_case}"
                    "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_partial_${input_case}.tsv"
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_partial_validation.cmake)
        set_tests_properties(${target_name}_partial_${input_case} PROPERTIES TIMEOUT 35 PROCESSORS 2)
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

# Optional Python tooling tests; building/running the C++ benchmarks does not require Python.
find_package(Python3 3.8 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME full_streaming_mpi
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/full_streaming_mpi_test.py
                --build-dir ${CMAKE_CURRENT_BINARY_DIR} --mpi-launcher ${MPIEXEC_EXECUTABLE})
    set_tests_properties(full_streaming_mpi PROPERTIES TIMEOUT 180 PROCESSORS 8)
    add_test(NAME full_roundoff_hv15r
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/full_roundoff_hv15r_test.py
                --build-dir ${CMAKE_CURRENT_BINARY_DIR} --mpi-launcher ${MPIEXEC_EXECUTABLE})
    set_tests_properties(full_roundoff_hv15r PROPERTIES TIMEOUT 300 PROCESSORS 32)
    add_test(NAME partial_cache_mpi
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/partial_cache_mpi_test.py
                --build-dir ${CMAKE_CURRENT_BINARY_DIR} --mpi-launcher ${MPIEXEC_EXECUTABLE})
    set_tests_properties(partial_cache_mpi PROPERTIES TIMEOUT 240 PROCESSORS 2)
    add_test(NAME benchmark_measurements
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_measurements_test.py
                --build-dir ${CMAKE_CURRENT_BINARY_DIR} --mpi-launcher ${MPIEXEC_EXECUTABLE}
                --numproc-flag=${MPIEXEC_NUMPROC_FLAG})
    set_tests_properties(benchmark_measurements PROPERTIES TIMEOUT 180 PROCESSORS 8)
    add_test(NAME matrix_tools
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/matrix_tools_test.py)
    set_tests_properties(matrix_tools PROPERTIES TIMEOUT 60)
    add_test(NAME experiment_scripts
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/experiment_scripts_test.py)
    set_tests_properties(experiment_scripts PROPERTIES TIMEOUT 240)
    add_test(NAME cresco_export
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cresco_export_test.py)
    set_tests_properties(cresco_export PROPERTIES TIMEOUT 60)
    add_test(NAME campaign
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/campaign_test.py)
    set_tests_properties(campaign PROPERTIES TIMEOUT 120)
endif()
