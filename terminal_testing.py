import time

import numpy as np

from utils.helping_methods import get_np_int_bit_string
from z_curve_computation.ConversionPrecompute import MortonConversionPrecompute
from z_curve_computation.z_curve_strings import z_index_string_manipulation, variables_string_manipulation
from z_curve_computation.z_curve_magic import generate_magic_number_arrays, z_index_magic_numbers, variables_magic_numbers
from z_curve_computation.z_curve_array_manipulation import variables_array_manipulation, z_index_array_manipulation


def magic_interleave_terminal_visualization():
    coordinate_np_type = np.uint8
    index_np_type = np.uint32
    var_count = 3
    variables = [5, 1, 4]

    B, S = generate_magic_number_arrays(coordinate_np_type, index_np_type, var_count)
    precomputed_vals = MortonConversionPrecompute(coordinate_np_type, index_np_type)

    z_index_final = z_index_string_manipulation(variables, coordinate_np_type, index_np_type, var_count, precomputed_vals)
    print(variables, "input variables")
    print(z_index_final, "directly computed morton number")
    v_s = np.asarray(variables, dtype=index_np_type)
    variable_strings = ["" for _ in range(var_count)]
    print()
    print("spread bits")
    for q in range(len(B) - 1, -1, -1):
        v_s = (v_s | (v_s << S[q])) & B[q]
        for k in range(var_count):
            variable_strings[k] += get_np_int_bit_string(v_s[k]).replace("0", ".")+"\n"
    for i in range(var_count):
        variable_strings[i] = variable_strings[i].strip()
        print("variable "+str(i))
        print(variable_strings[i])
    print()
    print("final process")
    for j in range(var_count):
        print(get_np_int_bit_string(v_s[j]).replace("0",".")," original variable "+str(j))
    print()
    final_number = np.dtype(index_np_type).type(0)
    for k in np.arange(var_count, dtype=index_np_type):
        value = v_s[k] << k
        final_number |= value
        print(get_np_int_bit_string(value).replace("0", "."), " shifted variable "+str(k))
    print(get_np_int_bit_string(final_number).replace("0", "."), " bitwise or (=morton number)")
    print(final_number, "resulting morton number")

def terminal_print_test():
    coordinate_np_type = np.uint16
    index_np_type = np.uint64

    val1 = 128
    val2 = 55555
    val3 = 0
    val4 = 1
    val5 = 1
    val6 = 1
    val7 = 1
    val8 = 1
    var_count = 4

    B, S = generate_magic_number_arrays(coordinate_np_type, index_np_type, var_count)
    precomputed_vals = MortonConversionPrecompute(coordinate_np_type, index_np_type)

    z_indexes_arg = np.array([val1, val2], dtype=index_np_type)

    print(z_indexes_arg, "z_indexes argument")
    variables_decoded = variables_array_manipulation(z_indexes_arg, coordinate_np_type, index_np_type, var_count, precomputed_vals)
    # variables_decoded_scalar = variables_magic_numbers(val1, coordinate_np_type, index_np_type, var_count, B, S)

    print(variables_decoded, "initially decoded variables")

    z_array_manipulation = z_index_array_manipulation(variables_decoded, coordinate_np_type, index_np_type, var_count, precomputed_vals)
    print(z_array_manipulation, "z_s array manipulation")

    z_string_manipulation = z_index_string_manipulation(variables_decoded, coordinate_np_type, index_np_type, var_count, precomputed_vals)
    print(z_string_manipulation, "z_s string manipulation")
    z_magic = z_index_magic_numbers(variables_decoded, index_np_type, B, S)
    print(z_magic, "z_s magic")