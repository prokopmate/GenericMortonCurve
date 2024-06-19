import math
import numpy as np


# methods using byte magic numbers, works in a general case
def z_index_magic_numbers(variables, index_type, B, S):
    v_s = np.asarray(variables, dtype=index_type)
    vs_ndim_minus_one = v_s.ndim-1
    for q in range(B.shape[0] - 1, -1, -1):
        v_s |= v_s << S[q]
        v_s &= B[q]
    shift_arrange = np.arange(v_s.shape[vs_ndim_minus_one], dtype=index_type)
    shifted_vs = v_s << shift_arrange.reshape((-1,)+v_s[0].shape)
    return np.bitwise_or.reduce(shifted_vs, axis=vs_ndim_minus_one, out=v_s[:, 0] if vs_ndim_minus_one > 0 else None)


def variables_magic_numbers(z_indexes_vars, coord_type, index_type, var_count, B, S):
    z_index_extended = np.asarray(z_indexes_vars, dtype=index_type)[..., np.newaxis]
    z_extended = (z_index_extended >> np.arange(var_count, dtype=index_type)) & B[0]
    for q in range(B.shape[0] - 1):
        z_extended |= z_extended >> S[q]
        z_extended &= B[q + 1]
    return z_extended.astype(coord_type, copy=False)


def generate_magic_number_arrays(coordinate_np_type, index_np_type, var_count):
    coordinate_bits_count = np.iinfo(coordinate_np_type).bits
    filled_len = var_count*coordinate_bits_count
    arrays_length = math.floor(math.log2(filled_len/var_count))
    powers_of_two = 2**(np.arange(arrays_length, dtype=index_np_type))
    bits_chunks = 2**powers_of_two-1
    calculation_shifts = var_count*powers_of_two
    S = calculation_shifts-powers_of_two
    B = np.zeros(arrays_length, dtype=index_np_type)
    for i in range(arrays_length):
        for q in np.arange(filled_len//calculation_shifts[i], dtype=index_np_type):
            B[i] |= bits_chunks[i] << q*calculation_shifts[i]
    return B, S
