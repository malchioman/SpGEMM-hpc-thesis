execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
            ${MPIEXEC_PREFLAGS} "${BENCHMARK}" ${MPIEXEC_POSTFLAGS}
            --matrix "${INPUT_MATRIX}"
            --threads 1 --warmup 0 --repeats 1 --trials 1
            --results "${RESULTS_FILE}"
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error_output
    TIMEOUT 20
)

if(NOT "${exit_code}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
        "Expected a nonzero exit code, got '${exit_code}'.\n${output}\n${error_output}")
endif()
if(NOT output MATCHES "(^|[\r\n])validation=FAIL([\r\n]|$)")
    message(FATAL_ERROR "Expected validation=FAIL.\n${output}\n${error_output}")
endif()

file(STRINGS "${RESULTS_FILE}" result_lines)
list(GET result_lines -1 last_result)
string(REPLACE "\t" ";" result_fields "${last_result}")
list(GET result_fields -1 validation_status)
if(NOT validation_status STREQUAL "FAIL")
    message(FATAL_ERROR "Expected FAIL in the last TSV row, got '${validation_status}'.")
endif()
