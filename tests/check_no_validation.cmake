cmake_minimum_required(VERSION 3.21)

if(INPUT_CASE STREQUAL "rectangular")
    set(input_args --matrix-a "${DATA_DIR}/tiny_rectangular.mtx"
                   --matrix-b "${DATA_DIR}/tiny_symmetric.mtx")
    set(expected_result "C=2x3 nnz=4")
    set(expected_c_nnz 4)
    set(validation_args --no-validate)
elseif(INPUT_CASE STREQUAL "overflow")
    set(input_args --no-validate --matrix-a "${DATA_DIR}/overflow_product.mtx")
    set(expected_result "C=2x2 nnz=2")
    set(expected_c_nnz 2)
elseif(INPUT_CASE STREQUAL "incompatible")
    set(input_args --matrix-a "${DATA_DIR}/tiny_symmetric.mtx"
                   --matrix-b "${DATA_DIR}/tiny_rectangular.mtx")
    set(validation_args --no-validate)
else()
    message(FATAL_ERROR "Unknown input case: ${INPUT_CASE}")
endif()

set(original_result_size 0)
if(EXISTS "${RESULTS_FILE}")
    file(SIZE "${RESULTS_FILE}" original_result_size)
endif()

execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
            ${MPIEXEC_PREFLAGS} "${BENCHMARK}" ${MPIEXEC_POSTFLAGS}
            ${input_args} --threads 1 --warmup 0 --repeats 1 --trials 1
            --results "${RESULTS_FILE}" ${validation_args}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error_output
    TIMEOUT 20
)

if(INPUT_CASE STREQUAL "incompatible")
    if(NOT "${exit_code}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR
            "Expected a nonzero exit code, got '${exit_code}'.\n${output}\n${error_output}")
    endif()
    string(FIND "${output}\n${error_output}" "SpGEMM requires A.cols == B.rows" error_position)
    if(error_position EQUAL -1)
        message(FATAL_ERROR "Expected dimension check failure.\n${output}\n${error_output}")
    endif()
    if(EXISTS "${RESULTS_FILE}")
        file(SIZE "${RESULTS_FILE}" result_size)
        if(NOT result_size EQUAL original_result_size)
            message(FATAL_ERROR "Incompatible inputs must not append a benchmark result.")
        endif()
    endif()
    return()
endif()

if(NOT "${exit_code}" STREQUAL "0")
    message(FATAL_ERROR "Expected success, got '${exit_code}'.\n${output}\n${error_output}")
endif()

string(REPLACE "\r\n" "\n" output "${output}")
foreach(expected_line IN ITEMS "validation=SKIPPED" "max_abs_error=NA" "${expected_result}")
    string(FIND "\n${output}" "\n${expected_line}\n" line_position)
    if(line_position EQUAL -1)
        message(FATAL_ERROR "Missing summary line '${expected_line}'.\n${output}")
    endif()
endforeach()

file(STRINGS "${RESULTS_FILE}" result_lines)
list(GET result_lines 0 header)
list(GET result_lines -1 last_result)
string(REPLACE "\t" ";" header_fields "${header}")
string(REPLACE "\t" ";" result_fields "${last_result}")
list(LENGTH header_fields header_count)
list(LENGTH result_fields result_count)
if(NOT result_count EQUAL header_count)
    message(FATAL_ERROR "TSV row does not match the header.")
endif()
set(expected_validation "SKIPPED")
set(expected_max_abs_error "NA")
foreach(field IN ITEMS validation max_abs_error c_nnz)
    list(FIND header_fields "${field}" field_index)
    if(field_index EQUAL -1)
        message(FATAL_ERROR "Missing TSV column '${field}'.")
    endif()
    list(GET result_fields ${field_index} actual_value)
    if(NOT "${actual_value}" STREQUAL "${expected_${field}}")
        message(FATAL_ERROR
            "TSV ${field}: expected '${expected_${field}}', got '${actual_value}'.")
    endif()
endforeach()
