/* ffnn.cpp */
#include <cmath>
#include <cstring>
#include <ffnn.h>
#include <mathlib.h>

namespace infinity {
namespace lm {

ffnn::ffnn()
{
    layer_number = 0;
    parameter_number = 0;
    activation_number = 0;
    parameter = nullptr;
    layer_size = nullptr;
    activation_function = nullptr;
}

ffnn::~ffnn()
{
    if (parameter != nullptr)
        delete[] parameter;

    if (layer_size != nullptr)
        delete[] layer_size;

    if (activation_function != nullptr)
        delete[] activation_function;
}

double* ffnn::get_parameter() const
{
    return parameter;
}

unsigned int ffnn::get_layer_number() const
{
    return layer_number;
}

unsigned int ffnn::get_parameter_number() const
{
    return parameter_number;
}

unsigned int ffnn::get_activation_number() const
{
    return activation_number;
}

unsigned int ffnn::get_layer_size(unsigned int n) const
{
    return layer_size[n];
}

activation_handler ffnn::get_activation_function(unsigned int i) const
{
    return activation_function[i];
}

double* ffnn::compute(double* act)
{
    double* layer = act;
    double* weight = parameter;
    activation_handler id_func = identity;

    *layer = 1.0;

    // forward
    for (unsigned int i = 0; i < layer_number - 1; i++) {
        unsigned int m = layer_size[i] + 1;
        unsigned int n = layer_size[i + 1];
        matrix_map a(layer + m + 1, n, 1);
        matrix_map w(weight, m, n);
        matrix_map x(layer, m, 1);
        auto act_func = activation_function[i + 1];

        *layer = 1.0;
        a.noalias() = w.transpose() * x;

        if (act_func != id_func)
            a = a.unaryExpr([act_func](double v) { return act_func(v); });

        layer += m;
        weight += m * n;
    }

    return layer + 1;
}

void ffnn::compute(double* in, double* act)
{
    double* layer = act;
    double* weight = parameter;
    activation_handler id_func = identity;

    std::memcpy(layer + 1, in, sizeof(double) * layer_size[0]);

    // forward
    for (unsigned int i = 0; i < layer_number - 1; i++) {
        unsigned int m = layer_size[i] + 1;
        unsigned int n = layer_size[i + 1];
        matrix_map a(layer + m + 1, n, 1);
        matrix_map w(weight, m, n);
        matrix_map x(layer, m, 1);
        auto act_func = activation_function[i + 1];

        a.noalias() = w.transpose() * x;

        if (act_func != id_func)
            a = a.unaryExpr([act_func](double v) { return act_func(v); });

        layer += m;
        weight += m * n;
    }
}

void ffnn::compute(double* in, double** neuron)
{
    double** layer = neuron;
    double* weight = parameter;
    activation_handler id_func = identity;

    std::memcpy(layer[0] + 1, in, sizeof(double) * layer_size[0]);

    for (unsigned int i = 0; i < layer_number - 1; i++) {
        unsigned int m = layer_size[i] + 1;
        unsigned int n = layer_size[i + 1];
        matrix_map a(layer[i + 1] + 1, n, 1);
        matrix_map w(weight, m, n);
        matrix_map x(layer[i], m, 1);
        auto act_func = activation_function[i + 1];

        a.noalias() = w.transpose() * x;

        if (act_func != id_func)
            a = a.unaryExpr([act_func](double v) { return act_func(v); });

        weight += m * n;
    }
}

void ffnn::initialize(unsigned int* size, unsigned int n)
{
    unsigned int pnum = 0;

    layer_number = n;
    layer_size = new unsigned int[n];
    activation_function = new activation_handler[n];
    // one bias neuron per layer
    activation_number = layer_number;

    for (unsigned int i = 0; i < n; i++) {
        layer_size[i] = size[i];
        activation_number += size[i];
        activation_function[i] = nullptr;
    }

    // calculate parameter number
    for (unsigned int i = 0; i < n - 1; i++) {
        pnum += (size[i] + 1) * size[i + 1];
    }

    // allocate memory
    parameter_number = pnum;
    parameter = new double[pnum];
}

void ffnn::set_activation_function(activation_handler f, unsigned int i)
{
    activation_function[i] = f;
}

} /* lm */
} /* infinity */
