/* nnltm.cpp */
#include <cmath>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <io.h>
#include <ffnn.h>
#include <cache.h>
#include <model.h>
#include <nnltm.h>
#include <vocab.h>
#include <mathlib.h>
#include <utility.h>
#include <parameter.h>

namespace infinity {
namespace lm {

nnltm::nnltm()
{
    flag = 0;
    window_size = 0;
    feature_number = 0;
    model_cache = nullptr;
    memory = nullptr;
    embedding = nullptr;
    input_vocab = nullptr;
    output_vocab = nullptr;
    neural_model = nullptr;
    output_function = nullptr;
}

nnltm::~nnltm()
{
    if (memory != nullptr)
        delete[] memory;

    if (model_cache != nullptr)
        delete model_cache;

    if (neural_model != nullptr)
        delete neural_model;
}

double nnltm::get_hit_rate() const
{
    return model_cache->get_hit_rate();
}

vocab* nnltm::get_input_vocab() const
{
    return input_vocab;
}

vocab* nnltm::get_output_vocab() const
{
    return output_vocab;
}

unsigned int nnltm::get_window_size() const
{
    return window_size;
}

void nnltm::precompute()
{
    ffnn* network = neural_model->get_network();
    unsigned int layer_number = network->get_layer_number();
    double* weight = network->get_parameter();
    activation_handler id_func = identity;
    unsigned int lookup_size;
    unsigned int input_layer_size;
    unsigned int hidden_layer_size;
    unsigned int layer_index;
    unsigned int fnum = feature_number;

    layer_index = 1;
    input_layer_size = network->get_layer_size(0);
    hidden_layer_size = network->get_layer_size(layer_index);
    lookup_size = hidden_layer_size;

    // for all hidden layers
    for (unsigned int i = layer_index; i < layer_number - 2; i++) {
        activation_handler func = network->get_activation_function(i);

        if (func != id_func)
            break;

        layer_index = i + 1;
        lookup_size = network->get_layer_size(layer_index);
    }

    // allocate memory
    memory = new double[(1 + input_number * window_size) * lookup_size];

    double* lookup_table = memory + lookup_size;

    // pre-compute lookup table
    for (unsigned int i = 0; i < input_number; i++) {
        // input embedding
        double* input = embedding + i * fnum;

        for (unsigned int j = 0; j < window_size; j++) {
            double* p = new double[hidden_layer_size];
            matrix_map wmat(weight, input_layer_size + 1, hidden_layer_size);

            // input layer -> hidden layer
            matrix_map xvec(input, fnum, 1);
            matrix_map yvec(p, hidden_layer_size, 1);
            auto w = wmat.block(1 + j * fnum, 0, fnum, hidden_layer_size);

            yvec.noalias() = w.transpose() * xvec;

            double* wmem = weight + (input_layer_size + 1) * hidden_layer_size;

            for (unsigned int k = 1; k < layer_index; k++) {
                unsigned int m = network->get_layer_size(k);
                unsigned int n = network->get_layer_size(k + 1);
                double* q = new double[n];
                matrix_map wmat(wmem, m + 1, n);
                matrix_map xvec(p, m, 1);
                matrix_map yvec(q, n, 1);
                auto w = wmat.block(1, 0, m, n);

                // hidden layer k -> hidden layer k + 1
                yvec.noalias() = w.transpose() * xvec;

                delete[] p;
                p = q;
                wmem += (m + 1) * n;
            }

            // copy to lookup table
            double* mem = lookup_table + (i * window_size + j) * lookup_size;

            for (unsigned int k = 0; k < lookup_size; k++) {
                mem[k] = p[k];
            }

            delete[] p;
        }
    }

    // pre-compute bias
    double* bias = memory;
    double *p = new double[hidden_layer_size];
    matrix_map wmat(weight, input_layer_size + 1, hidden_layer_size);
    matrix_map ymat(p, hidden_layer_size, 1);

    ymat = wmat.block(0, 0, 1, hidden_layer_size).transpose();

    double* wmem = weight + (input_layer_size + 1) * hidden_layer_size;

    for (unsigned int i = 1; i < layer_index; i++) {
        unsigned int m = network->get_layer_size(i);
        unsigned int n = network->get_layer_size(i + 1);
        double* q = new double[n];
        matrix_map wmat(wmem, m + 1, n);
        matrix_map xmat(p, m, 1);
        matrix_map ymat(q, n, 1);

        // previous bias * weight
        ymat.noalias() = wmat.block(1, 0, m, n).transpose() * xmat;
        // add new bias weight
        ymat += wmat.block(0, 0, 1, n).transpose();

        delete[] p;
        p = q;
        wmem += (m + 1) * n;
    }

    // copy to memory
    for (unsigned int i = 0; i < lookup_size; i++)
        bias[i] = p[i];

    delete[] p;

    // set flag
    flag = layer_index;
}

void nnltm::load(const char* name)
{
    neural_model = new model;

    load_model(name, neural_model);
    parameter* param = neural_model->get_parameter();
    ffnn* network = neural_model->get_network();
    embedding = neural_model->get_embedding();
    activation_number = network->get_activation_number();

    window_size = neural_model->get_window();
    input_number = neural_model->get_input_size();
    output_number = neural_model->get_output_size();
    feature_number = neural_model->get_feature_size();
    input_vocab = neural_model->get_target_vocab();
    output_vocab = neural_model->get_output_vocab();

    std::string model;
    std::string out_func;

    param->get_parameter("model", model);
    param->get_parameter("output-function", out_func);

    if (model != "nnltm") {
        std::string msg = "incompatible model, must be nnltm";
        throw std::runtime_error(msg);
    }

    if (out_func == "softmax")
        output_function = softmax;
    else
        output_function = identity;

    model_cache = new cache;

    set_thread_number(1);
}

void nnltm::set_cache_size(unsigned int n)
{
    model_cache->resize(n);
}

double nnltm::probability(unsigned int* input)
{
    double score;
    double* layer;
    unsigned int* context = input;
    unsigned int label = input[window_size];
    ffnn* network = neural_model->get_network();
    double* parameter = network->get_parameter();
    double* activation;
    unsigned int layer_number = network->get_layer_number();
    double* weight = parameter;
    activation_handler id_func = identity;

    mutex.lock();
    double* result = model_cache->find(input, window_size + 1);
    score = (result == nullptr) ? 0.0 : *result;
    mutex.unlock();

    if (result != nullptr) {
        return score;
    }

    activation = new double[activation_number];
    layer = activation;

    if (flag) {
        unsigned int m;
        unsigned int n;
        unsigned int start_layer = flag;

        // skip linear layers
        for (unsigned int i = 0; i < start_layer; i++) {
            m = network->get_layer_size(i);

            layer += m + 1;
        }

        // skip linear layers
        for (unsigned int i = 0; i < start_layer - 1; i++) {
            m = network->get_layer_size(i);
            n = network->get_layer_size(i + 1);

            weight += (m + 1) * n;
        }

        m = network->get_layer_size(start_layer - 1);
        n = network->get_layer_size(start_layer);

        double* lookup_bias = memory;
        double* lookup_table = memory + n;
        matrix_map init_layer(layer + 1, n, 1);
        matrix_map init_bias(lookup_bias, n, 1);
        auto init_func = network->get_activation_function(start_layer);

        init_layer = init_bias;

        for (unsigned int i = 0; i < window_size; i++) {
            unsigned int pos = i;
            unsigned int ind = input[i];
            double* data = lookup_table + ind * window_size * n + pos * n;
            matrix_map data_map(data, n, 1);
            init_layer += data_map;
        }

        auto lambda = [init_func](double v) { return init_func(v); };
        init_layer.noalias() = init_layer.unaryExpr(lambda);
        weight += (m + 1) * n;

        // hidden layers
        for (unsigned int i = start_layer; i < layer_number - 2; i++) {
            unsigned int m = network->get_layer_size(i) + 1;
            unsigned int n = network->get_layer_size(i + 1);
            auto act_func = network->get_activation_function(i + 1);
            matrix_map a(layer + m + 1, n, 1);
            matrix_map w(weight, m, n);
            matrix_map x(layer, m, 1);

            *layer = 1.0;
            a.noalias() = w.transpose() * x;

            if (act_func != id_func)
                a = a.unaryExpr([act_func](double v) { return act_func(v); });

            layer += m;
            weight += m * n;
        }
    } else {
        // input layer, index to word vector
        for (unsigned int i = 0; i < window_size; i++) {
            unsigned int id = context[i];
            double* vec = embedding + id * feature_number;
            double* ptr = layer + 1 + i * feature_number;

            for (unsigned int j = 0; j < feature_number; j++)
                ptr[j] = vec[j];
        }

        // for all hidden layers
        for (unsigned int i = 0; i < layer_number - 2; i++) {
            unsigned int m = network->get_layer_size(i) + 1;
            unsigned int n = network->get_layer_size(i + 1);
            auto act_func = network->get_activation_function(i + 1);
            matrix_map a(layer + m + 1, n, 1);
            matrix_map w(weight, m, n);
            matrix_map x(layer, m, 1);

            *layer = 1.0;
            a.noalias() = w.transpose() * x;

            if (act_func != id_func)
                a = a.unaryExpr([act_func](double v) { return act_func(v); });

            layer += m;
            weight += m * n;
        }
    }

    // output layer
    if (output_function == softmax) {
        unsigned int m = network->get_layer_size(layer_number - 2) + 1;
        unsigned int n = network->get_layer_size(layer_number - 1);
        auto act_func = network->get_activation_function(layer_number - 1);
        matrix_map a(layer + m + 1, n, 1);
        matrix_map w(weight, m, n);
        matrix_map x(layer, m, 1);
        double* output_layer = layer + m + 1;

        // need calculate all elements in output layer, very slow
        *layer = 1.0;
        a.noalias() = w.transpose() * x;

        if (act_func != id_func)
            a = a.unaryExpr([act_func](double v) { return act_func(v); });

        output_function(output_layer, n);

        score = output_layer[label];
    } else {
        unsigned int m = network->get_layer_size(layer_number - 2) + 1;
        unsigned int n = network->get_layer_size(layer_number - 1);
        auto act_func = network->get_activation_function(layer_number - 1);
        matrix_map a(layer + m + 1, n, 1);
        matrix_map w(weight, m, n);
        matrix_map x(layer, m, 1);
        auto wvec = w.col(label);
        auto aval = a.block(label, 0, 1, 1);

        *layer = 1.0;
        aval.noalias() = wvec.transpose() * x;

        score = aval(0, 0);
        score = act_func(score);
    }

    mutex.lock();
    model_cache->update(input, window_size + 1, score);
    mutex.unlock();

    delete[] activation;

    return score;
}

} /* lm */
} /* infinity */
