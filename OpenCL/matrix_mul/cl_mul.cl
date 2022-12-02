R"(

// First naive implementation
__kernel void naive_mul(const unsigned int lhs_rows,  
                        const unsigned int rhs_rows, 
                        const unsigned int rhs_cols, const __global int* lhs,
                                                     const __global int* rhs,
                                                           __global int* res) 
{
    // Thread identifiers
    const int global_row = get_global_id(0); // Row ID of C (0..M)
    const int global_col = get_global_id(1); // Col ID of C (0..N)
 

    // Compute a single element (loop over rhs_rows)
    int mul_res = 0;
    for (int com_size = 0; com_size < rhs_rows; ++com_size) 

        mul_res += lhs[com_size + rhs_rows * global_row] * rhs[global_col + rhs_cols*com_size];
    
 
    // Store the result
    res[global_row * rhs_cols + global_col] = mul_res;
}
)"