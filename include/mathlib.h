/*
 * mathlib.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __MATHLIB_H__
#define __MATHLIB_H__

#include <handler.h>
#include <external/Eigen/Dense>

namespace infinity {
namespace lm {

typedef Eigen::Map<Eigen::Matrix<double, -1, -1>> matrix_map;

double sigmoid(double v);
double identity(double v);
void identity(double* v, unsigned int n);
void tanh(double* v, unsigned int n);
void softmax(double* v, unsigned int n);

double tanh_derivative(double a);
double sigmoid_derivative(double a);
double identity_derivative(double a);
derivative_handler derivative(activation_handler f);

double log_cost(double* a, unsigned int i, unsigned int n);
double nce_cost(double* a, unsigned int i, unsigned int n);
double entropy_cost(double* a, unsigned int i, unsigned int n);
double sn_cost(double* a, double p, unsigned int i, unsigned int n);

void vecxmat(double* d, double* v, double* a, unsigned int m, unsigned int n);
void matxvec(double* d, double* a, double* v, unsigned int m, unsigned int n);
void addmat(double* a, double* c, double* r, unsigned int m, unsigned int n);
double dotprod(double* v1, double* v2, unsigned int n);

} /* lm */
} /* infinity */

#endif /* __MATHLIB_H__ */
