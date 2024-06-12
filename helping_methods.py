import numpy as np


# methods to get bits string from np.uint integers
def get_np_int_bit_string(int_val):
    return np.binary_repr(int_val, width=np.iinfo(int_val.dtype).bits)


def get_np_int_bit_string_width_arg(int_val, width):
    return np.binary_repr(int_val, width=width)


def get_nice_np_bin_string(int_val, length_val):
    bin_repr = get_np_int_bit_string(int_val)
    str_val=""
    for i in range(length_val//8):
        str_val += bin_repr[i*8:8+i*8]+" "
    return str_val


# get variables from classic index on grid
def classic_nd_variables(z, mod_val, n_dim):
    return np.array(np.unravel_index(z, [mod_val for _ in range(n_dim)])).T
    # return np.array([z%mod_val,z//mod_val])
