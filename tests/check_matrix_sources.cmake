if(INPUT_CASE STREQUAL "reuse_a")
    set(expected_a "${DATA_DIR}/tiny_symmetric.mtx")
    set(expected_b "${expected_a}")
    set(expected_shape "3x3")
    set(input_args --matrix-a "${expected_a}")
elseif(INPUT_CASE STREQUAL "explicit_b")
    set(expected_a "${DATA_DIR}/tiny_rectangular.mtx")
    set(expected_b "${DATA_DIR}/tiny_symmetric.mtx")
    set(expected_shape "2x3")
    set(input_args --matrix-a "${expected_a}" --matrix-b "${expected_b}")
elseif(INPUT_CASE STREQUAL "synthetic")
    set(expected_a "synthetic")
    set(expected_b "synthetic")
    set(expected_shape "8x5")
    set(input_args --rows 8 --cols 8 --b-cols 5 --nnz-per-row 2 --b-nnz-per-row 2)
else()
    message(FATAL_ERROR "Unknown input case: ${INPUT_CASE}")
endif()

file(REMOVE "${RESULTS_FILE}")

execute_process(
    COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
            ${MPIEXEC_PREFLAGS} "${BENCHMARK}" ${MPIEXEC_POSTFLAGS}
            ${input_args} --threads 1 --warmup 0 --repeats 1 --trials 1
            --results "${RESULTS_FILE}"
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error_output
    TIMEOUT 20
)
if(NOT "${exit_code}" STREQUAL "0")
    message(FATAL_ERROR "Expected success, got '${exit_code}'.\n${output}\n${error_output}")
endif()

string(REPLACE "\r\n" "\n" output "${output}")
foreach(expected_line IN ITEMS "matrix_a_source=${expected_a}" "matrix_b_source=${expected_b}"
                               "validation=PASS")
    string(FIND "\n${output}" "\n${expected_line}\n" line_position)
    if(line_position EQUAL -1)
        message(FATAL_ERROR "Missing summary line '${expected_line}'.\n${output}")
    endif()
endforeach()
string(FIND "\n${output}" "\nC=${expected_shape} nnz=" shape_position)
if(shape_position EQUAL -1)
    message(FATAL_ERROR "Expected result shape ${expected_shape}.\n${output}")
endif()

file(STRINGS "${RESULTS_FILE}" result_lines)
list(GET result_lines 0 header)
list(GET result_lines -1 last_result)
string(REPLACE "\t" ";" header_fields "${header}")
string(REPLACE "\t" ";" result_fields "${last_result}")
set(expected_matrix_a_source "${expected_a}")
set(expected_matrix_b_source "${expected_b}")
set(expected_validation "PASS")
foreach(field IN ITEMS matrix_a_source matrix_b_source validation)
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
