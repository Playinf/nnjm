/* mathlib.cpp */
#include <cmath>
#include <limits>
#include <mathlib.h>

namespace infinity {
namespace lm {

double sigmoid(double v)
{
    return 1.0 / (1.0 + std::exp(-v));
}

double identity(double v)
{
    return v;
}

void identity(double* v, unsigned int n)
{
    return;
}

void tanh(double* v, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
        v[i] = std::tanh(v[i]);
}

void softmax(double* v, unsigned int n)
{
    double max = -std::numeric_limits<double>::infinity();
    double norm = 0.0;

    // safe softmax
    for (unsigned int i = 0; i < n; i++) {
        if (v[i] > max)
            max = v[i];
    }

    for (unsigned int i = 0; i < n; i++) {
        double t = std::exp(v[i] - max);
        norm += t;
        v[i] = t;
    }

    for (unsigned int i = 0; i < n; i++) {
        v[i] /= norm;
    }
}

double tanh_derivative(double a)
{
    return 1 - a * a;
}

double sigmoid_derivative(double a)
{
    return a * (1 - a);
}

double identity_derivative(double a)
{
    return 1.0;
}

derivative_handler derivative(activation_handler f)
{
    if (f == sigmoid)
        return sigmoid_derivative;
    else if (f == static_cast<activation_handler>(std::tanh))
        return tanh_derivative;
    else if (f == static_cast<activation_handler>(identity))
        return identity_derivative;
    else
        return nullptr;
}

double log_cost(double* a, unsigned int i, unsigned int n)
{
    return -std::log(a[i]);
}

double nce_cost(double* a, unsigned int i, unsigned int n)
{
    return sn_cost(a, 1.0, i, n);
}

double entropy_cost(double* a, unsigned int lab, unsigned int n)
{
    double cost = 0;

    for (unsigned int i = 0; i < n; i++) {
        double val = a[i];

        if (lab == i)
            cost += -std::log(val);
        else
            cost += -std::log(1 - val);
    }

    return cost;
}

double sn_cost(double* a, double p, unsigned int i, unsigned int n)
{
    double sum = 0.0;
    double factor;

    for (unsigned int i = 0; i < n; i++) {
        sum += std::exp(a[i]);
    }

    // log(Z(x))
    factor = std::log(sum);

    return -(a[i] - factor - p * factor * factor);
}

// d: a n dimensional vector
// v: a m dimensional vector
// a: a mn dimensional matrix
void vecxmat(double* d, double* v, double* a, unsigned int m, unsigned int n)
{
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        double sum = 0.0f;

        for (unsigned int j = 0; j < m; j++) {
            sum += v[j] * a[i + j * n];
        }

        d[i] = sum;
    }
}

void matxvec(double* d, double* a, double* v, unsigned int m, unsigned int n)
{
    for (unsigned int i = 0; i < m; i++) {
        double* w = a + n * i;
        d[i] = dotprod(w, v, n);
    }
}

// a: a mn dimensional matrix
// c: a m dimensional column vector
// r: a n dimensional row vector
void addmat(double* a, double* c, double* r, unsigned int m, unsigned int n)
{
    double* p = a;

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++)
            *p++ += c[i] * r[j];
    }
}

double dotprod(double* v1, double* v2, unsigned int n)
{
    double sum = 0.0f;

    for (unsigned int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }

    return sum;
}

} /* lm */
} /* infinity */
