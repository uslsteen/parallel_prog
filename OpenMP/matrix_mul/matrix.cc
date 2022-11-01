#include "mul_impl.hh"
#include "timer.hh"

template <typename Data>
void Buffer_init(Data &buffer, uint &num_of_rws, uint &num_of_clmns)
{
    std::cin >> num_of_rws >> num_of_clmns;
    buffer.reserve(num_of_rws * num_of_clmns);
    //
    for (size_t i = 0; i < num_of_rws * num_of_clmns; ++i)
        std::cin >> buffer[i];
}

void check(mul_optimiz::Matr_int &lhs, const mul_optimiz::Matr_int &rhs)
{
    if (lhs == rhs)
        std::cout << "Equal\n";
    else
        std::cout << "Not equal\n";
}

int main()
{
    uint lhs_rows = 0, lhs_cols = 0, rhs_rows = 0, rhs_cols = 0;
    std::vector<int> lhs_buf, rhs_buf, np_buf;
    //
    Buffer_init(lhs_buf, lhs_rows, lhs_cols);
    Buffer_init(rhs_buf, rhs_rows, rhs_cols);
    Buffer_init(np_buf, lhs_rows, rhs_cols);
    //
    mul_optimiz::Matr_int lhs{lhs_rows, lhs_cols, lhs_buf};
    mul_optimiz::Matr_int rhs{rhs_rows, rhs_cols, rhs_buf};
    mul_optimiz::Matr_int np_ref{lhs_rows, rhs_cols, np_buf};

    //! NOTE: Naive multiplication
    auto start = omp_get_wtime();
    auto naive_res = lhs * rhs;
    auto end = omp_get_wtime();
    std::cout << "Naive time : " << (end - start) * 1000 << std::endl;
    //
    check(np_ref, naive_res);

    //! NOTE: OpenMP parallel mul
    start = omp_get_wtime();
    auto openmp_res = mul_optimiz::openmp_impl::naive_mul(lhs, rhs);
    end = omp_get_wtime();
    //
    std::cout << "OpenMP time : " << (end - start) * 1000 << std::endl;

    //! NOTE: vinograd linear mul
    auto vinograd_linear_res = mul_optimiz::vinograd_mul(lhs, rhs);
    check(np_ref, vinograd_linear_res);

    //! NOTE: vinograd linear mul
    start = omp_get_wtime();
    auto vinograd_parallel_res = mul_optimiz::openmp_impl::vinograd_mul(lhs, rhs);
    //
    check(np_ref, vinograd_parallel_res);
//
#if 0
    //! NOTE: OpenMP parallel mul
    double start = omp_get_wtime();
    auto openmp_res = mul_optimiz::openmp_impl::naive_mul(lhs, rhs);
    double end = omp_get_wtime();
    //
    std::cout << "OpenMP time : " <<  end - start << std::endl;

    //! NOTE: naive mul
    start = omp_get_wtime();
    auto naive_res = lhs * rhs;
    end = omp_get_wtime();
    //
    std::cout << "Naive time : " <<  end - start << std::endl;
    check(naive_res, openmp_res);
    
    //! NOTE: simp vectorization mul
    start = omp_get_wtime();
    auto simd_res = mul_optimiz::simd_impl::mul(lhs, rhs);
    end = omp_get_wtime();
    //
    std::cout << "SIMD time : " <<  end - start << std::endl;
    check(naive_res, simd_res);

    //! NOTE: OpenMP SIMD directive mul
    start = omp_get_wtime();
    auto open_simd_res = mul_optimiz::openmp_impl::naive_mul(lhs, rhs, true);
    end = omp_get_wtime();
    //
    std::cout << "OpenMP SIMD time : " <<  end - start << std::endl;
#endif
    return 0;
}
