# GenericMortonCurve
 Methods to interleave and deinterleave Morton curve (z-order) in a generic case, written in Python
 
 -> Conversion can be tested in the main script with terminal testing functions or visualisation functions




:Implementation of 3 types of methods to interleave and deinterleave morton curve in a generic case defined by coordinate type, index type and variables count. Methods support both scalar and vector conversion.

-> magic_numbers methods - works using magic binary numbers (B,S arrays) precomputed for specific types, works for general case, well performing method especially insteresting for my future cpp implementation

-> string_manipulation methods - easy to understand but worse performing methods, works in general case

-> array manipulation methods - uses numpy manipulations of bit arrays, works only if variables_count==index_bits_count/coordinate_bits_count, well perfoming method




:Generic visualisation of classic index and morton curve plots in 2-d and 3-d cases->included in curves_comparison_2d and curves_comparison_3d functions




:Generic visualisation of histogram of i to i+1 distances in variable space for classic n-d index and n-d morton curve index ->included in distances_comparison function
