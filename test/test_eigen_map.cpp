#include <Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
typedef Eigen::Map<matrix> matrix_map;

int main()
{
    double a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    matrix_map mat(a, 3, 4);
    auto bmat = mat.block(0, 1, 3, 3);

    std::cout << mat << std::endl;
    std::cout << bmat << std::endl;
    bmat(0, 0) = 0;
    std::cout << mat << std::endl;
    std::cout << bmat << std::endl;
}
