import numpy as np


class MortonConversionPrecompute:
    def __init__(self, coord_type, index_type):
        is_little_byteorder = lambda b_order_str: (b_order_str == "<" or (b_order_str == "=" and np.little_endian)
                                                   or b_order_str == '|')

        coord_type_byte_order = np.dtype(coord_type).byteorder
        index_type_byte_order = np.dtype(index_type).byteorder
        index_little = is_little_byteorder(index_type_byte_order)
        if coord_type_byte_order == "|":
            coord_type_byte_order = index_type_byte_order
        coord_little = is_little_byteorder(coord_type_byte_order)
        self.index_order = 2 * index_little - 1
        self.coord_order = 2 * coord_little - 1
        self.coord_bit_order = 'little' if coord_little else 'big'
        self.index_bit_order = 'little' if index_little else 'big'
        self.swapping_bytes = coord_little != index_little
