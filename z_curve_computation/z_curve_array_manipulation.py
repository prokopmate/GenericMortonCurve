import numpy as np


# methods using numpy array manipulation, works only if variable_count==index_bits_count//coordinate_bits_count
def z_index_array_manipulation(variables, coord_type, index_type, var_count, precomputed_vals):
    coordinate_bits_count = np.iinfo(coord_type).bits
    variables_transposed = np.asarray(variables, dtype=coord_type).T[::precomputed_vals.index_order]
    if precomputed_vals.swapping_bytes:
        variables_transposed = variables_transposed.byteswap()
    z_array_len = np.size(variables_transposed[0])
    variables_bytes = variables_transposed.tobytes()
    variables_bits = np.unpackbits(np.frombuffer(variables_bytes, dtype=np.uint8), bitorder=precomputed_vals.index_bit_order)
    interleaved_bits = variables_bits.reshape(var_count, coordinate_bits_count*z_array_len).T
    packed_index_bytes = np.packbits(interleaved_bits, bitorder=precomputed_vals.index_bit_order)
    return np.squeeze(np.frombuffer(packed_index_bytes, dtype=index_type))


def variables_array_manipulation(z_index_arr, coord_type, index_type, var_count, precomputed_vals):
    coordinate_bits_count = np.iinfo(coord_type).bits
    z_array_len = np.size(z_index_arr)
    z_index_arr_idx_type = np.asarray(z_index_arr,dtype=index_type)
    if precomputed_vals.swapping_bytes:
        z_index_arr_idx_type = z_index_arr_idx_type.byteswap()
    value_bytes = z_index_arr_idx_type.tobytes()
    bits = np.unpackbits(np.frombuffer(value_bytes, dtype=np.uint8), bitorder=precomputed_vals.coord_bit_order)
    conversion_bits_array = bits.reshape((coordinate_bits_count*z_array_len, var_count)).T
    result_bytes = np.packbits(conversion_bits_array, bitorder=precomputed_vals.coord_bit_order)
    flattened_variable_array = np.frombuffer(result_bytes, dtype=coord_type)
    variable_array = flattened_variable_array.reshape((var_count, z_array_len))[::precomputed_vals.coord_order].T
    return np.squeeze(variable_array)