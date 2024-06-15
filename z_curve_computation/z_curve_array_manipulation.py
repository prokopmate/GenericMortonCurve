import numpy as np


# methods using numpy array manipulation, works only if variable_count==index_bits_count//coordinate_bits_count
def z_index_array_manipulation(variables, coord_type, index_type, var_count):
    bigger_endian_index_type = np.dtype(index_type).newbyteorder('>')
    coordinate_bits_count = np.iinfo(coord_type).bits
    variables_transposed = coord_type(variables).T
    z_array_len = np.size(variables_transposed[0])
    if variables_transposed.dtype.byteorder != ">" and (variables_transposed.dtype.byteorder != "=" or np.little_endian):
        variables_transposed = variables_transposed.byteswap()
    variables_bytes = variables_transposed.tobytes()
    variables_bits = np.unpackbits(np.frombuffer(variables_bytes, dtype=np.uint8))
    interleaved_bits = variables_bits.reshape(var_count, coordinate_bits_count*z_array_len).T
    packed_index_bytes = np.packbits(interleaved_bits)
    return np.squeeze(np.frombuffer(packed_index_bytes, dtype=bigger_endian_index_type))


def variables_array_manipulation(z_index_arr, coord_type, index_type, var_count):
    bigger_endian_coord_type = np.dtype(coord_type).newbyteorder('>')
    coordinate_bits_count = np.iinfo(coord_type).bits
    z_array_len = np.size(z_index_arr)
    z_index_arr_idx_type = index_type(z_index_arr)
    if z_index_arr_idx_type.dtype.byteorder != ">" and (z_index_arr_idx_type.dtype.byteorder != "=" or np.little_endian):
        z_index_arr_idx_type = z_index_arr_idx_type.byteswap()
    value_bytes = z_index_arr_idx_type.tobytes()
    bits = np.unpackbits(np.frombuffer(value_bytes, dtype=np.uint8))
    conversion_bits_array = bits.reshape((coordinate_bits_count*z_array_len, var_count)).T
    result_bytes = np.packbits(conversion_bits_array)
    flattened_variable_array = np.frombuffer(result_bytes, dtype=bigger_endian_coord_type)
    variable_array = flattened_variable_array.reshape((var_count, z_array_len)).T
    return np.squeeze(variable_array)