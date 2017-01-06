#include <ctime>
#include <iostream>
#include <functional>
#include <handler.h>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, -1, -1> col_matrix;
typedef Eigen::Map<Eigen::Matrix<double, -1, -1>> col_matrix_map;
typedef Eigen::Matrix<double, -1, -1, 1> row_matrix;
typedef Eigen::Map<Eigen::Matrix<double, -1, -1, 1>> row_matrix_map;

void test_row_mat()
{
    row_matrix wmat(513, 32000);
    row_matrix xmat(128, 513);
    row_matrix ymat(128, 32000);

    auto t1 = std::clock();

    for (unsigned int i = 0; i < 100; i++) {
        ymat.noalias() = xmat * wmat;
    }

    auto t2 = std::clock();

    std::cerr << 100.0 / ((double) (t2 - t1) / CLOCKS_PER_SEC) << std::endl;
}

void test_col_mat()
{
    col_matrix wmat(513, 32000);
    col_matrix xmat(128, 513);
    col_matrix ymat(128, 32000);

    auto t1 = std::clock();

    for (unsigned int i = 0; i < 100; i++) {
        ymat.noalias() = xmat * wmat;
    }

    auto t2 = std::clock();

    std::cerr << 100.0 / ((double) (t2 - t1) / CLOCKS_PER_SEC) << std::endl;
}

void test_col_mat_trans()
{
    col_matrix wmat(513, 32000);
    col_matrix xmat(513, 128);
    col_matrix ymat(32000, 128);

    auto t1 = std::clock();

    for (unsigned int i = 0; i < 100; i++) {
        ymat.noalias() = wmat.transpose() * xmat;
    }

    auto t2 = std::clock();

    std::cerr << 100.0 / ((double) (t2 - t1) / CLOCKS_PER_SEC) << std::endl;
}

int main()
{
    test_row_mat();
    test_col_mat();
    test_col_mat_trans();
}
