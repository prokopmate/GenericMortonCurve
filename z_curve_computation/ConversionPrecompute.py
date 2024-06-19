import numpy as np


class MortonConversionPrecompute:
    def __init__(self, coord_type, index_type):
        coord_type_byte_order = np.dtype(coord_type).byteorder
        index_type_byte_order = np.dtype(index_type).byteorder
        coord_little = coord_type_byte_order == "<" or (coord_type_byte_order == "=" and np.little_endian)
        index_little = index_type_byte_order == "<" or (index_type_byte_order == "=" and np.little_endian)
        self.index_order = 2 * index_little - 1
        self.coord_order = 2 * coord_little - 1
        self.coord_bit_order = 'little' if coord_little else 'big'
        self.index_bit_order = 'little' if index_little else 'big'
        self.swapping_bytes = coord_little != index_little
