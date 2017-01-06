/* nnjm.cpp */
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <io.h>
#include <ffnn.h>
#include <nnjm.h>
#include <cache.h>
#include <model.h>
#include <vocab.h>
#include <mathlib.h>
#include <utility.h>
#include <parameter.h>

namespace infinity {
namespace lm {

nnjm::nnjm()
{
    order = 0;
    start_layer = 0;
    source_context = 0;
    target_context = 0;
    feature_number = 0;
    jm_cache = nullptr;
    embedding = nullptr;
    lookup_bias = nullptr;
    lookup_table = nullptr;
    source_vocab = nullptr;
    target_vocab = nullptr;
    output_vocab = nullptr;
    neural_model = nullptr;
    output_function = nullptr;
}

nnjm::~nnjm()
{
    if (jm_cache != nullptr)
        delete jm_cache;

    if (lookup_bias != nullptr)
        delete[] lookup_bias;

    if (lookup_table != nullptr)
        delete[] lookup_table;

    if (neural_model != nullptr)
        delete neural_model;
}

double nnjm::get_hit_rate() const
{
    return jm_cache->get_hit_rate();
}

unsigned int nnjm::get_order() const
{
    return order;
}

vocab* nnjm::get_source_vocab() const
{
    return source_vocab;
}

vocab* nnjm::get_target_vocab() const
{
    return target_vocab;
}

vocab* nnjm::get_output_vocab() const
{
    return output_vocab;
}

unsigned int nnjm::get_source_context() const
{
    return source_context;
}

unsigned int nnjm::get_target_context() const
{
    return target_context;
}

void nnjm::precompute()
{
    ffnn* network = neural_model->get_network();
    unsigned int layer_number = network->get_layer_number();
    double* weight = network->get_parameter();
    activation_handler id_func = identity;
    unsigned int lookup_size;
    unsigned int input_layer_size;
    unsigned int first_hidden_size;

    start_layer = 1;
    input_layer_size = network->get_layer_size(0);
    lookup_size = network->get_layer_size(start_layer);
    first_hidden_size = lookup_size;

    // for all hidden layers
    for (unsigned int i = start_layer; i < layer_number - 2; i++) {
        activation_handler func = network->get_activation_function(i);

        if (func != id_func)
            break;

        start_layer = i + 1;
        lookup_size = network->get_layer_size(start_layer);
    }

    // bias unit
    double* p;
    double* q;
    double* w;

    p = new double[first_hidden_size];

    matrix_map ymat(p, first_hidden_size, 1);
    matrix_map first_weight(weight, input_layer_size + 1, first_hidden_size);

    ymat = first_weight.block(0, 0, 1, first_hidden_size).transpose();

    w = weight + (input_layer_size + 1) * first_hidden_size;

    for (unsigned int i = 1; i < start_layer; i++) {
        unsigned int m = network->get_layer_size(i);
        unsigned int n = network->get_layer_size(i + 1);
        matrix_map wmat(w, m + 1, n);
        matrix_map xmat(p, m, 1);

        q = new double[n];
        matrix_map ymat(q, n, 1);

        ymat.noalias() = wmat.block(1, 0, m, n).transpose() * xmat;
        ymat += wmat.block(0, 0, 1, n).transpose();

        delete[] p;
        p = q;
    }

    lookup_bias = p;

    // allocate memory
    lookup_table = new double[input_number * lookup_size * (order - 1)];

    // do multiply
    for (unsigned int i = 0; i < input_number; i++) {
        // input embedding
        double* input = embedding + i * feature_number;
        unsigned int l0 = input_layer_size;
        unsigned int l1 = first_hidden_size;
        double* init_weight = weight;

        for (unsigned int j = 0; j < order - 1; j++) {
            double* p = new double[first_hidden_size];
            double* wmem = init_weight;
            double* memory = lookup_table + i * lookup_size * (order - 1);
            matrix_map wmat(wmem, input_layer_size + 1, first_hidden_size);
            auto w = wmat.block(1 + j * feature_number, 0, feature_number, l1);

            memory += j * lookup_size;

            // input layer -> hidden layer
            matrix_map xvec(input, feature_number, 1);
            matrix_map yvec(p, first_hidden_size, 1);

            yvec.noalias() = w.transpose() * xvec;
            wmem += (l0 + 1) * l1;

            for (unsigned int k = 1; k < start_layer; k++) {
                unsigned int m = network->get_layer_size(k);
                unsigned int n = network->get_layer_size(k + 1);
                double* q = new double[n];
                matrix_map wmat(wmem, m + 1, n);
                matrix_map xvec(p, m, 1);
                matrix_map yvec(q, n, 1);
                auto wb = wmat.block(1, 0, m, n);

                yvec.noalias() = wb.transpose() * xvec;

                delete[] p;
                p = q;
            }

            // copy to lookup table
            for (unsigned int k = 0; k < lookup_size; k++) {
                memory[k] = p[k];
            }

            delete[] p;
        }
    }
}

void nnjm::load(const char* name)
{
    neural_model = new model;

    load_model(name, neural_model);
    parameter* param = neural_model->get_parameter();
    ffnn* network = neural_model->get_network();
    embedding = neural_model->get_embedding();
    activation_number = network->get_activation_number();

    order = neural_model->get_order();
    input_number = neural_model->get_input_size();
    output_number = neural_model->get_output_size();
    feature_number = neural_model->get_feature_size();
    source_context = neural_model->get_source_context();
    target_context = neural_model->get_target_context();
    source_vocab = neural_model->get_source_vocab();
    target_vocab = neural_model->get_target_vocab();
    output_vocab = neural_model->get_output_vocab();

    std::string model;
    std::string out_func;

    param->get_parameter("model", model);
    param->get_parameter("output-function", out_func);

    if (model != "nnjm") {
        std::string msg = "incompatible model, must be nnjm";
        throw std::runtime_error(msg);
    }

    if (out_func == "softmax")
        output_function = softmax;
    else
        output_function = identity;

    jm_cache = new cache;

    set_thread_number(1);
}

void nnjm::set_cache_size(unsigned int n)
{
    jm_cache->resize(n);
}

double nnjm::ngram_prob(unsigned int* input)
{
    double score;
    double* layer;
    unsigned int* context = input;
    unsigned int label = input[order - 1];
    ffnn* network = neural_model->get_network();
    double* parameter = network->get_parameter();
    double* activation;
    unsigned int layer_number = network->get_layer_number();
    double* weight = parameter;
    activation_handler id_func = identity;

    mutex.lock();
    double* result = jm_cache->find(input, order);
    mutex.unlock();

    if (result != nullptr)
        return *result;

    activation = new double[activation_number];
    layer = activation;

    if (start_layer) {
        unsigned int m;
        unsigned int n;

        for (unsigned int i = 0; i < start_layer; i++) {
            m = network->get_layer_size(i);

            layer += m + 1;
        }

        for (unsigned int i = 0; i < start_layer - 1; i++) {
            m = network->get_layer_size(i);
            n = network->get_layer_size(i + 1);

            weight += (m + 1) * n;
        }

        m = network->get_layer_size(start_layer - 1);
        n = network->get_layer_size(start_layer);
        matrix_map init_layer(layer + 1, n, 1);
        matrix_map init_bias(lookup_bias, m, 1);
        auto init_func = network->get_activation_function(start_layer);

        init_layer = init_bias;

        for (unsigned int i = 0; i < order - 1; i++) {
            unsigned int pos = i;
            unsigned int ind = input[i];
            double* data = lookup_table + ind * (order - 1) * n + pos * n;
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
        for (unsigned int i = 0; i < order - 1; i++) {
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
    jm_cache->update(input, order, score);
    mutex.unlock();

    delete[] activation;

    return score;
}

double nnjm::probability(unsigned int* input)
{
    double score = 0.0;
    unsigned int* context = input;
    unsigned int label = input[order - 1];
    ffnn* network = neural_model->get_network();
    double* parameter = network->get_parameter();
    unsigned int layer_number = network->get_layer_number();
    activation_handler id_func = identity;

    mutex.lock();
    double* result = jm_cache->find(input, order);
    mutex.unlock();

    if (result != nullptr)
        return *result;

    double* activation = new double[activation_number];
    double* layer = activation;
    double* weight = parameter;

    // input layer, index to word vector
    for (unsigned int i = 0; i < order - 1; i++) {
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
        double* act = layer + m + 1;
        double* wmat = weight;
        double* x = layer;

        *layer = 1.0;
        vecxmat(act, x, wmat, n, m);

        if (act_func != id_func) {
            for (unsigned int j = 0; j < n; j++)
                act[j] = act_func(act[j]);
        }

        layer += m;
        weight += m * n;
    }

    unsigned int m = network->get_layer_size(layer_number - 2) + 1;
    unsigned int n = network->get_layer_size(layer_number - 1);
    auto act_func = network->get_activation_function(layer_number - 1);

    *layer = 1.0;

    for (unsigned int i = 0; i < m; i++) {
        score += layer[i] * weight[i * n + label];
    }

    score = act_func(score);

    mutex.lock();
    jm_cache->update(input, order, score);
    mutex.unlock();

    delete[] activation;

    return score;
}

} /* lm */
} /* infinity */
