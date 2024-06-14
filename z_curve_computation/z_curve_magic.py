import math
import numpy as np


# methods using byte magic numbers, works in a general case
def z_index_magic_numbers(variables, index_type, var_count, B, S):
    v_s = variables.astype(index_type).T
    z_s = np.zeros_like(v_s[0], dtype=index_type)
    for q in range(len(B) - 1, -1, -1):
        v_s = (v_s | (v_s << S[q])) & B[q]
    for i in range(len(v_s)):
        vs_shifted = v_s[var_count - i - 1] << index_type(i)
        z_s = z_s | vs_shifted
    return z_s


def variables_magic_numbers(z_indexes_vars, coord_type, index_type, var_count, B, S):
    z_index_extended = index_type(z_indexes_vars)[..., np.newaxis]
    z_extended = (z_index_extended >> np.arange(var_count - 1, -1, -1, dtype=index_type)) & B[0]
    for q in range(len(B) - 1):
        z_extended |= z_extended >> S[q]
        z_extended &= B[q + 1]
    return coord_type(z_extended)


def generate_magic_number_arrays(coordinate_np_type, index_np_type, var_count):
    coordinate_bits_count = np.iinfo(coordinate_np_type).bits
    filled_len = index_np_type(var_count*coordinate_bits_count)
    arrays_length = math.floor(math.log2(filled_len/var_count))
    powers_of_two = np.array([2**i for i in range(arrays_length)], dtype=index_np_type)
    bits_chunks = 2**powers_of_two-1
    calculation_shifts = var_count*powers_of_two
    S = (var_count-1)*powers_of_two
    B = np.zeros(arrays_length, dtype=index_np_type)
    for i in range(arrays_length):
        for q in range(filled_len//calculation_shifts[i]):
            B[i] |= bits_chunks[i] << index_np_type(q*calculation_shifts[i])
    return B, S
