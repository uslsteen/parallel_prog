#include <iostream>
#include <vector>
#include <string>
#include <cmath>


const double PI = std::atan(1) * 4;

//! NOTE: x belongs (0, 1]
//! NOTE: t belongs (0, 1]
constexpr double T = 1,
                 X = 1;

//! NOTE: step along the x and t axes
constexpr double h = 1e-2,
                 tau = 1e-2;

//! NOTE: num of steps
constexpr double x_steps = X / h,
                 t_steps = T / tau;

inline double phi(double x) {
    return std::cos(PI * x);
}

inline double psi(double t) {
    return std::exp(-t);
}

inline double f(double x, double t) {
    return x + t;
}

int main() {

    

    return 0;
}