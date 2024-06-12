import numpy as np


# methods using numpy array manipulation, works only if variable_count==index_bits_count//coordinate_bits_count
def z_index_array_manipulation(variables, coord_type, index_type, var_count):
    coordinate_bits_count = np.iinfo(coord_type).bits
    z_array_len = len(variables)
    variables_bytes = variables.T.tobytes() if variables.dtype.byteorder == ">" else variables.byteswap().T.tobytes()
    variables_bits = np.unpackbits(np.frombuffer(variables_bytes, dtype=np.uint8))
    interleaved_bits = variables_bits.reshape(var_count, coordinate_bits_count*z_array_len).T
    packed_index_bytes = np.packbits(interleaved_bits)
    return np.frombuffer(packed_index_bytes, dtype=np.dtype(index_type).newbyteorder('>'))


def variables_array_manipulation(z_index_arr, coord_type, index_type, var_count):
    coordinate_bits_count = np.iinfo(coord_type).bits
    index_bits_count = np.iinfo(index_type).bits
    z_array_len = len(z_index_arr)
    final_string_filled_len = var_count*coordinate_bits_count*z_array_len
    value_bytes = z_index_arr.tobytes() if z_index_arr.dtype.byteorder == ">" else z_index_arr.byteswap().tobytes()
    bits = np.unpackbits(np.frombuffer(value_bytes, dtype=np.uint8))
    bits_start_index = index_bits_count*z_array_len-final_string_filled_len
    relevant_bits = bits[bits_start_index:]
    conversion_bits_array = relevant_bits.reshape((coordinate_bits_count*z_array_len, var_count)).T
    result_bytes = np.packbits(conversion_bits_array)
    flattened_variable_array = np.frombuffer(result_bytes, dtype=np.dtype(coord_type).newbyteorder('>'))
    variable_array = flattened_variable_array.reshape(var_count, z_array_len).T.astype(coord_type)
    return variable_array



