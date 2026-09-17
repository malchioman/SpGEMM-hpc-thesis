cmake_minimum_required(VERSION 3.21)

set(bad_rows 1e3)
set(bad_threads 2junk)
set(bad_warmup 0suffix)
file(REMOVE "${RESULTS_FILE}")

foreach(option IN ITEMS rows threads warmup)
    execute_process(
        COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
                ${MPIEXEC_PREFLAGS} "${BENCHMARK}" ${MPIEXEC_POSTFLAGS}
                --rows 16 --cols 16 --b-cols 16 --nnz-per-row 1 --b-nnz-per-row 1
                --threads 1 --warmup 0 --repeats 1 --trials 1
                --results "${RESULTS_FILE}" "--${option}" "${bad_${option}}"
        RESULT_VARIABLE exit_code
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error_output
        TIMEOUT 20
    )
    if(NOT "${exit_code}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR
            "Expected rejection of --${option} ${bad_${option}}, got '${exit_code}'.\n${output}\n${error_output}")
    endif()
    if(option STREQUAL "warmup")
        set(expected "--warmup expects a non-negative integer")
    else()
        set(expected "--${option} expects a positive integer")
    endif()
    string(FIND "${output}\n${error_output}" "${expected}" error_position)
    if(error_position EQUAL -1)
        message(FATAL_ERROR "Missing numeric argument diagnostic.\n${output}\n${error_output}")
    endif()
    if(EXISTS "${RESULTS_FILE}")
        message(FATAL_ERROR "Invalid numeric arguments must not write benchmark results.")
    endif()
endforeach()
