execute_process(
    COMMAND "${MPIEXEC}" "${NUMPROC_FLAG}" 2 ${MPI_PREFLAGS} "${PROGRAM}" ${MPI_POSTFLAGS}
            --matrix "${MATRIX}" --threads 1 --warmup 0 --repeats 1 --trials 1
            --results "${RESULTS}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
    TIMEOUT 30
)
if(NOT status MATCHES "^[0-9]+$" OR status EQUAL 0 OR NOT output MATCHES "validation=FAIL")
    message(FATAL_ERROR "Expected a validation failure and nonzero exit, got ${status}\n${output}\n${errors}")
endif()
file(READ "${RESULTS}" report)
if(NOT report MATCHES "FAIL\tprepared_halo_v2")
    message(FATAL_ERROR "Failed validation was not recorded in the TSV")
endif()
