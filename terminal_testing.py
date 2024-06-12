import numpy as np
from z_curve_computation.z_curve_strings import z_index_string_manipulation, variables_string_manipulation
from z_curve_computation.z_curve_magic import generate_magic_number_arrays, z_index_magic_numbers, variables_magic_numbers
from z_curve_computation.z_curve_array_manipulation import variables_array_manipulation, z_index_array_manipulation


def terminal_print_test():
    coordinate_np_type = np.uint16
    index_np_type = np.uint64

    val1 = coordinate_np_type(255)
    val2 = coordinate_np_type(255)
    val3 = coordinate_np_type(0)
    val4 = coordinate_np_type(1)
    val5 = coordinate_np_type(1)
    val6 = coordinate_np_type(1)
    val7 = coordinate_np_type(1)
    val8 = coordinate_np_type(1)
    var_count = 4

    B, S = generate_magic_number_arrays(coordinate_np_type, index_np_type, var_count)

    z_indexes_arg = np.array([val1, val2], dtype=index_np_type)

    print(z_indexes_arg, "z_indexes argument")
    variables_decoded = variables_magic_numbers(z_indexes_arg, coordinate_np_type, index_np_type, var_count, B, S)
    print(variables_decoded, "initially decoded variables")

    z_array_manipulation = z_index_array_manipulation(variables_decoded, coordinate_np_type, index_np_type, var_count)
    print(z_array_manipulation, "z_s array manipulation")

    print(z_index_string_manipulation(variables_decoded, coordinate_np_type, index_np_type, var_count),
          "z_s string manipulation")
    z_magic = z_index_magic_numbers(variables_decoded, index_np_type, var_count, B, S)

    print(z_magic, "z_s magic")