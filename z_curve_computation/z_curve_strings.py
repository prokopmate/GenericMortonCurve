import numpy as np
from utils.helping_methods import get_np_int_bit_string_width_arg,get_np_int_bit_string


# methods using python string manipulation, works in a general case
def z_index_string_manipulation(variables_array, coord_type, index_type, var_count):
    reversed_index_type = np.dtype(index_type).newbyteorder('>')  # ensures correct order for conversion
    coordinate_bits_count = np.iinfo(coord_type).bits
    index_bits_count = np.iinfo(index_type).bits
    final_string_filled_len = var_count * coordinate_bits_count
    padding_zeroes = "0" * max(0, index_bits_count - final_string_filled_len)

    z_index_arr = []
    for variables in np.atleast_2d(variables_array):
        bit_strings = [get_np_int_bit_string_width_arg(var,coordinate_bits_count) for var in variables]
        final_bit_string = ""
        for i in range(final_string_filled_len):
            final_bit_string += bit_strings[i % var_count][i//var_count]
        final_bit_string = padding_zeroes+final_bit_string
        result_bytes = np.packbits([int(c_bit) for c_bit in final_bit_string])
        z_index_arr.append(np.frombuffer(result_bytes, dtype=reversed_index_type)[0])
    return index_type(np.squeeze(z_index_arr))


def variables_string_manipulation(z_index_arr, coord_type, index_type, var_count):
    reversed_coord_type = np.dtype(coord_type).newbyteorder('>')  # ensures correct order for conversion
    coordinate_bits_count = np.iinfo(coord_type).bits
    index_bits_count = np.iinfo(index_type).bits
    final_string_filled_len = var_count * coordinate_bits_count

    variables_arr = []
    for z_index in np.atleast_1d(index_type(z_index_arr)):
        z_index_str = get_np_int_bit_string(z_index)
        var_strings = ["" for _ in range(var_count)]
        start_index = index_bits_count-final_string_filled_len
        for i in range(final_string_filled_len):
            var_strings[i % var_count] += z_index_str[start_index+i]
        variable_array = [np.uint8(var_strings[i//coordinate_bits_count][i % coordinate_bits_count]) for i in range(final_string_filled_len)]
        result_bytes = np.packbits(variable_array)
        variables_arr.append(np.frombuffer(result_bytes, dtype=reversed_coord_type))
    return np.squeeze(variables_arr)

