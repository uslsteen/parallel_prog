#include "mul_impl.hh"

template <typename Data>
void Buffer_init(Data& buffer, uint& num_of_rws, uint& num_of_clmns)
{
    std::cin >> num_of_rws >> num_of_clmns;
    buffer.reserve(num_of_rws * num_of_clmns);
    //
    for (size_t i = 0; i < num_of_rws*num_of_clmns; ++i)
        std::cin >> buffer[i];
}

void check(mul_optimiz::Matr_int& lhs, const mul_optimiz::Matr_int& rhs) {
    if (lhs == rhs)
        std::cout << "Equal\n";
    else
        std::cout << "Not equal\n";
}

int main() {
    //
    std::cout << "n dev" << omp_get_num_devices() << std::endl;
    std::cout << "Num of threads : " << omp_get_num_threads() << std::endl;

    uint lhs_rows = 0, lhs_cols = 0, rhs_rows = 0, rhs_cols = 0;
    std::vector<int> lhs_buf, rhs_buf;

    Buffer_init(lhs_buf, lhs_rows, lhs_cols);
    Buffer_init(rhs_buf, rhs_rows, rhs_cols);
    //
    mul_optimiz::Matr_int lhs{lhs_rows, lhs_cols, lhs_buf};
    mul_optimiz::Matr_int rhs{rhs_rows, rhs_cols, rhs_buf};
    
    //! NOTE: OpenMP parallel mul
    double start = omp_get_wtime();
    auto openmp_res = mul_optimiz::openmp_impl::mul(lhs, rhs);
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
    auto open_simd_res = mul_optimiz::openmp_impl::mul(lhs, rhs, true);
    end = omp_get_wtime();
    //
    std::cout << "OpenMP SIMD time : " <<  end - start << std::endl;
    return 0;
}
