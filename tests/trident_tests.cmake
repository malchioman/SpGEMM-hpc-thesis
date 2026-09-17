add_executable(trident_correctness tests/trident_correctness.cpp)
target_link_libraries(trident_correctness PRIVATE trident_two_sided_support)
enable_project_warnings(trident_correctness)

add_executable(trident_hybrid_correctness tests/trident_correctness.cpp)
target_compile_definitions(trident_hybrid_correctness PRIVATE TRIDENT_HYBRID)
target_link_libraries(trident_hybrid_correctness PRIVATE trident_hybrid_support)
enable_project_warnings(trident_hybrid_correctness)

add_executable(trident_get_correctness tests/trident_correctness.cpp)
target_compile_definitions(trident_get_correctness PRIVATE TRIDENT_GET)
target_link_libraries(trident_get_correctness PRIVATE trident_get_support Threads::Threads)
enable_project_warnings(trident_get_correctness)

add_executable(trident_get_lifecycle tests/trident_get_lifecycle.cpp)
target_link_libraries(trident_get_lifecycle PRIVATE trident_get_support Threads::Threads)
enable_project_warnings(trident_get_lifecycle)

foreach(layout IN ITEMS 1x1 1x3 2x1 2x2 3x1)
    if(layout STREQUAL "1x1")
        set(ranks 1)
        set(node_size 1)
    elseif(layout STREQUAL "1x3")
        set(ranks 3)
        set(node_size 3)
    elseif(layout STREQUAL "2x1")
        set(ranks 4)
        set(node_size 1)
    elseif(layout STREQUAL "2x2")
        set(ranks 8)
        set(node_size 2)
    else()
        set(ranks 9)
        set(node_size 1)
    endif()
    add_test(NAME trident_correctness_${layout}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_correctness> ${MPIEXEC_POSTFLAGS} ${node_size})
    set_tests_properties(trident_correctness_${layout} PROPERTIES TIMEOUT 90 PROCESSORS ${ranks})
    add_test(NAME trident_hybrid_correctness_${layout}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_hybrid_correctness> ${MPIEXEC_POSTFLAGS} ${node_size})
    set_tests_properties(trident_hybrid_correctness_${layout} PROPERTIES TIMEOUT 90 PROCESSORS ${ranks})
    add_test(NAME trident_get_correctness_${layout}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_get_correctness> ${MPIEXEC_POSTFLAGS} ${node_size})
    set_tests_properties(trident_get_correctness_${layout} PROPERTIES TIMEOUT 90 PROCESSORS ${ranks})
    add_test(NAME trident_get_lifecycle_${layout}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_get_lifecycle> ${MPIEXEC_POSTFLAGS} ${node_size})
    set_tests_properties(trident_get_lifecycle_${layout} PROPERTIES TIMEOUT 90 PROCESSORS ${ranks})
endforeach()

add_test(NAME trident_correctness_large_messages
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_correctness> ${MPIEXEC_POSTFLAGS} 2 large)
set_tests_properties(trident_correctness_large_messages PROPERTIES TIMEOUT 90 PROCESSORS 8)

foreach(mode IN ITEMS large skew)
    add_test(NAME trident_hybrid_correctness_${mode}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_hybrid_correctness> ${MPIEXEC_POSTFLAGS} 2 ${mode})
    set_tests_properties(trident_hybrid_correctness_${mode} PROPERTIES TIMEOUT 90 PROCESSORS 8)
    add_test(NAME trident_get_correctness_${mode}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_get_correctness> ${MPIEXEC_POSTFLAGS} 2 ${mode})
    set_tests_properties(trident_get_correctness_${mode} PROPERTIES TIMEOUT 90 PROCESSORS 8)
endforeach()

foreach(input_case IN ITEMS hierarchical invalid_grid invalid_logical)
    add_test(NAME trident_cli_${input_case}
        COMMAND ${CMAKE_COMMAND}
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DBENCHMARK=$<TARGET_FILE:trident_two_sided>" "-DINPUT_CASE=${input_case}"
            "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/trident_${input_case}.tsv"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_trident_cli.cmake)
    set_tests_properties(trident_cli_${input_case} PROPERTIES TIMEOUT 60 PROCESSORS 8)
    add_test(NAME trident_hybrid_cli_${input_case}
        COMMAND ${CMAKE_COMMAND}
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DBENCHMARK=$<TARGET_FILE:trident_hybrid>" "-DINPUT_CASE=${input_case}"
            "-DEXPECTED_PROTOCOL=trident_hybrid_rma_requests_v1"
            "-DEXPECTED_INTER_TRANSPORT=hybrid_rma_requests_two_sided"
            "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/trident_hybrid_${input_case}.tsv"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_trident_cli.cmake)
    set_tests_properties(trident_hybrid_cli_${input_case} PROPERTIES TIMEOUT 60 PROCESSORS 8)
    add_test(NAME trident_get_cli_${input_case}
        COMMAND ${CMAKE_COMMAND}
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DBENCHMARK=$<TARGET_FILE:trident_get>" "-DINPUT_CASE=${input_case}"
            "-DEXPECTED_PROTOCOL=trident_get_csr_v1"
            "-DEXPECTED_INTER_TRANSPORT=one_sided_get"
            "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/trident_get_${input_case}.tsv"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_trident_cli.cmake)
    set_tests_properties(trident_get_cli_${input_case} PROPERTIES TIMEOUT 60 PROCESSORS 8)
endforeach()
