import numpy as np
from utils.helping_methods import get_np_int_bit_string_width_arg,get_np_int_bit_string


# methods using python string manipulation, works in a general case
def z_index_string_manipulation(variables_array, coord_type, index_type, var_count, precomputed_vals):
    coordinate_bits_count = np.iinfo(coord_type).bits
    index_bits_count = np.iinfo(index_type).bits
    final_string_filled_len = var_count * coordinate_bits_count
    at_least_2d_vars = np.atleast_2d(np.asarray(variables_array, dtype=coord_type))
    z_index_arr = np.empty(len(at_least_2d_vars), dtype=index_type)
    padding_zeroes = "0" * max(0, index_bits_count - final_string_filled_len)
    var_count_minus_one = var_count-1
    for index, variables in enumerate(at_least_2d_vars):
        bit_strings = [get_np_int_bit_string_width_arg(var, coordinate_bits_count) for var in variables]
        final_bit_string = ""
        for i in range(final_string_filled_len)[::-precomputed_vals.index_order]:
            final_bit_string += bit_strings[var_count_minus_one-i % var_count][i//var_count]
        final_bit_string = "".join((padding_zeroes, final_bit_string,)[::-precomputed_vals.index_order])
        result_bytes = np.packbits([int(c_bit) for c_bit in final_bit_string], bitorder=precomputed_vals.index_bit_order)
        z_index_arr[index] = np.frombuffer(result_bytes, dtype=index_type)[0]
    return np.squeeze(z_index_arr)


def variables_string_manipulation(z_index_arr, coord_type, index_type, var_count, precomputed_vals):
    coordinate_bits_count = np.iinfo(coord_type).bits
    index_bits_count = np.iinfo(index_type).bits
    final_string_filled_len = var_count * coordinate_bits_count
    z_index_arr_at_least_1d = np.atleast_1d(np.asarray(z_index_arr, dtype=index_type))
    variables_arr = np.empty((len(z_index_arr_at_least_1d), var_count), dtype=coord_type)
    var_count_minus_one = var_count-1
    for index, z_index in enumerate(z_index_arr_at_least_1d):
        z_index_str = get_np_int_bit_string(z_index)
        var_strings = ["" for _ in range(var_count)]
        start_index = index_bits_count-final_string_filled_len
        for i in range(final_string_filled_len)[::-precomputed_vals.coord_order]:
            var_strings[var_count_minus_one-i % var_count] += z_index_str[start_index+i]
        variable_array = [np.uint8(var_strings[i//coordinate_bits_count][i % coordinate_bits_count]) for i in range(final_string_filled_len)]
        result_bytes = np.packbits(variable_array, bitorder=precomputed_vals.coord_bit_order)
        variables_arr[index] = np.frombuffer(result_bytes, dtype=coord_type)
    return np.squeeze(variables_arr)

