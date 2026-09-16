add_executable(trident_correctness tests/trident_correctness.cpp)
target_link_libraries(trident_correctness PRIVATE trident_two_sided_support)
enable_project_warnings(trident_correctness)

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
endforeach()

add_test(NAME trident_correctness_large_messages
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:trident_correctness> ${MPIEXEC_POSTFLAGS} 2 large)
set_tests_properties(trident_correctness_large_messages PROPERTIES TIMEOUT 90 PROCESSORS 8)

foreach(input_case IN ITEMS hierarchical invalid_grid invalid_logical)
    add_test(NAME trident_cli_${input_case}
        COMMAND ${CMAKE_COMMAND}
            "-DMPIEXEC_EXECUTABLE=${MPIEXEC_EXECUTABLE}" "-DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
            "-DMPIEXEC_PREFLAGS=${MPIEXEC_PREFLAGS}" "-DMPIEXEC_POSTFLAGS=${MPIEXEC_POSTFLAGS}"
            "-DBENCHMARK=$<TARGET_FILE:trident_two_sided>" "-DINPUT_CASE=${input_case}"
            "-DRESULTS_FILE=${CMAKE_CURRENT_BINARY_DIR}/trident_${input_case}.tsv"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_trident_cli.cmake)
    set_tests_properties(trident_cli_${input_case} PROPERTIES TIMEOUT 60 PROCESSORS 8)
endforeach()
