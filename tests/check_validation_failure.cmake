cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED EXPECTED_PROTOCOL)
    set(EXPECTED_PROTOCOL prepared_halo_v2)
endif()
file(REMOVE "${RESULTS}")

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
file(STRINGS "${RESULTS}" lines)
list(GET lines 0 header)
list(GET lines -1 row)
string(REPLACE "\t" ";" fields "${header}")
string(REPLACE "\t" ";" values "${row}")
set(expected_validation FAIL)
set(expected_benchmark_protocol "${EXPECTED_PROTOCOL}")
foreach(field IN ITEMS validation benchmark_protocol)
    list(FIND fields "${field}" index)
    if(index LESS 0)
        message(FATAL_ERROR "Missing ${field} in TSV")
    endif()
    list(GET values ${index} actual)
    if(NOT "${actual}" STREQUAL "${expected_${field}}")
        message(FATAL_ERROR "TSV ${field}: expected ${expected_${field}}, got ${actual}")
    endif()
endforeach()
