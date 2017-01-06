/* trainer.cpp */
#include <map>
#include <cmath>
#include <cstring>
#include <functional>
#include <ffnn.h>
#include <model.h>
#include <mathlib.h>
#include <trainer.h>
#include <utility.h>
#include <parameter.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_lock_t int
#define omp_init_lock(lock) ;
#define omp_set_lock(lock) ;
#define omp_unset_lock(lock) ;
#define omp_destroy_lock(lock) ;
#define omp_get_max_threads() 1
#endif

namespace infinity {
namespace lm {

trainer::trainer()
{
    queue_size = 0;
    layer_number = 0;
    error = nullptr;
    neuron = nullptr;
    synapse = nullptr;
    gradient = nullptr;
    embedding = nullptr;
    offset = nullptr;
    queue = nullptr;
    layer_size = nullptr;
    network = nullptr;
    neural_model = nullptr;
    error_function = nullptr;
    activation_function = nullptr;

    control.batch_size = 1;
    control.learning_rate = 1.0;
}

trainer::~trainer()
{
    if (error != nullptr)
        delete[] error;

    if (neuron != nullptr)
        delete[] neuron;

    if (gradient != nullptr)
        delete[] gradient;

    if (offset != nullptr)
        delete[] offset;

    if (queue != nullptr)
        delete[] queue;

    if (error_function != nullptr)
        delete error_function;

    if (activation_function != nullptr)
        delete[] activation_function;
}

double* trainer::get_error(unsigned int i) const
{
    return error + offset[i].layer_offset;
}

double* trainer::get_layer(unsigned int i) const
{
    return neuron + offset[i].layer_offset;
}

double* trainer::get_weight(unsigned int i) const
{
    return synapse + offset[i].weight_offset;
}

double* trainer::get_gradient(unsigned int i) const
{
    return gradient + offset[i].gradient_offset;
}

unsigned int trainer::get_size(unsigned int i) const
{
    return layer_size[i];
}

void trainer::initialize(ffnn* network)
{
    unsigned int nnum = 0;
    unsigned int pnum = 0;
    unsigned int bnum = 0;

    this->network = network;
    layer_number = network->get_layer_number();
    synapse = network->get_parameter();

    offset = new offset_type[layer_number];
    layer_size = new unsigned int[layer_number];
    activation_function = new activation_handler[layer_number];

    unsigned int n = layer_number;

    // copy layer size
    for (unsigned int i = 0; i < n; i++) {
        layer_size[i] = network->get_layer_size(i);
        activation_function[i] = network->get_activation_function(i);
    }

    // calculate neuron number and parameter number
    for (unsigned int i = 0; i < n; i++)
        nnum += layer_size[i] + 1;

    for (unsigned int i = 0; i < n - 1; i++)
        pnum += (layer_size[i] + 1) * layer_size[i + 1];

    bnum = control.batch_size;
    // allocate memory
    error = new double[nnum * bnum];
    neuron = new double[nnum * bnum];
    gradient = new double[pnum];

    // initialize neuron
    double* p = neuron;

    for (unsigned int i = 0; i < n; i++) {
        unsigned int num = layer_size[i] + 1;
        matrix_map neu(p, num, bnum);
        // bias unit
        neu.row(0) = matrix_map::Constant(1, bnum, 1.0);
        p += num * bnum;
    }

    // calculate offset
    offset[0].layer_offset = 0;
    offset[0].weight_offset = 0;
    offset[0].gradient_offset = 0;

    for (unsigned int i = 1; i < n; i++) {
        unsigned int num = layer_size[i - 1] + 1;
        offset[i].layer_offset = offset[i - 1].layer_offset + num * bnum;
    }

    for (unsigned int i = 1; i < n - 1; i++) {
        unsigned int inum = layer_size[i - 1] + 1;
        unsigned int onum = layer_size[i];
        unsigned int num = inum * onum;
        offset[i].weight_offset = offset[i - 1].weight_offset + num;
        offset[i].gradient_offset = offset[i - 1].gradient_offset + num;
    }
}

void trainer::initialize(model* m)
{
    unsigned int nnum = 0;
    unsigned int pnum = 0;

    neural_model = m;
    embedding = m->get_embedding();
    network = neural_model->get_network();
    layer_number = network->get_layer_number();
    offset = new offset_type[layer_number];
    layer_size = new unsigned int[layer_number];
    activation_function = new activation_handler[layer_number];
    synapse = network->get_parameter();

    unsigned int n = layer_number;

    // copy layer size and activation function
    for (unsigned int i = 0; i < n; i++) {
        layer_size[i] = network->get_layer_size(i);
        activation_function[i] = network->get_activation_function(i);
    }

    // calculate neuron number and parameter number
    for (unsigned int i = 0; i < n; i++)
        nnum += layer_size[i] + 1;

    for (unsigned int i = 0; i < n - 1; i++)
        pnum += (layer_size[i] + 1) * layer_size[i + 1];

    unsigned int bnum = control.batch_size;
    unsigned int order = m->get_order();
    unsigned int fsize = m->get_feature_size();
    unsigned int qsize = (order - 1) * bnum;
    unsigned int gradient_size = pnum + qsize * fsize;

    // allocate memory
    error = new double[nnum * bnum];
    neuron = new double[nnum * bnum];
    queue = new unsigned int[qsize];
    gradient = new double[gradient_size];

    std::memset(gradient, 0, sizeof(double) * gradient_size);

    // initialize neuron
    double* p = neuron;

    for (unsigned int i = 0; i < n; i++) {
        unsigned int num = layer_size[i] + 1;
        matrix_map neu(p, num, bnum);
        // bias unit
        neu.row(0) = matrix_map::Constant(1, bnum, 1.0);
        p += num * bnum;
    }

    // calculate offset
    offset[0].layer_offset = 0;
    offset[0].weight_offset = 0;
    offset[0].gradient_offset = 0;

    for (unsigned int i = 1; i < n; i++) {
        unsigned int num = layer_size[i - 1] + 1;
        offset[i].layer_offset = offset[i - 1].layer_offset + num * bnum;
    }

    for (unsigned int i = 1; i < n - 1; i++) {
        unsigned int inum = layer_size[i - 1] + 1;
        unsigned int onum = layer_size[i];
        unsigned int num = inum * onum;
        offset[i].weight_offset = offset[i - 1].weight_offset + num;
        offset[i].gradient_offset = offset[i - 1].gradient_offset + num;
    }

    // gradient for word embeddings
    offset[n - 1].gradient_offset = pnum;
}

void trainer::compute(double* in)
{
    network->compute(in, neuron);
}

void trainer::update_parameter()
{
    double alpha = control.learning_rate;

    // update network parameter
    for (unsigned int i = 0; i < layer_number - 1; i++) {
        double* weight = get_weight(i);
        double* gradient = get_gradient(i);
        unsigned int inum = get_size(i) + 1;
        unsigned int onum = get_size(i + 1);
        matrix_map delta(gradient, inum, onum);
        matrix_map weightmat(weight, inum, onum);

        delta *= alpha;
        delta = delta.unaryExpr(gradient_constraint);
        weightmat -= delta;
        weightmat = weightmat.unaryExpr(parameter_constraint);
    }

    // update word embedding
    for (unsigned int i = 0; i < queue_size; i++) {
        unsigned int id = queue[i];
        double* gvec = get_gradient(layer_number -1);
        unsigned int fsize = neural_model->get_feature_size();
        double* vec = embedding + id * fsize;
        matrix_map gmat(gvec, fsize, queue_size);
        matrix_map feamat(vec, fsize, 1);
        auto delta = gmat.col(i);

        delta *= alpha;
        delta = delta.unaryExpr(gradient_constraint);

        feamat -= delta;
        feamat = feamat.unaryExpr(parameter_constraint);
    }

    queue_size = 0;
}

double trainer::train(unsigned int* x)
{
    return train(x, 1);
}

double trainer::train(double* x, unsigned int y)
{
    return train(x, &y, 1);
}

// training model, batch version
double trainer::train(unsigned int* input, unsigned int num)
{
    unsigned int order = neural_model->get_order();
    unsigned int feature_size = neural_model->get_feature_size();
    unsigned int* label = new unsigned int[num];

    // index to word vector
    for (unsigned int i = 0; i < num; i++) {
        unsigned int* context = input + i * order;
        matrix_map layer(get_layer(0), layer_size[0] + 1, num);
        auto neu = layer.block(1, 0, layer_size[0], num);
        auto neuvec = neu.col(i);

        for (unsigned int j = 0; j < order - 1; j++) {
            unsigned int id = context[j];
            double* vec = embedding + id * feature_size;
            matrix_map feature(vec, feature_size, 1);
            auto invec = neuvec.block(j * feature_size, 0, feature_size, 1);

            invec = feature;
        }

        label[i] = context[order - 1];
    }

    double cost = backpropagation(label, num);

    // calculate gradient
    for (unsigned int i = 0; i < layer_number - 1; i++) {
        unsigned int m = layer_size[i] + 1;
        unsigned int n = layer_size[i + 1];
        double* avec = get_layer(i);
        double* g = get_gradient(i);
        double* err = get_error(i + 1);
        matrix_map grad(g, m, n);
        matrix_map errvec(err, n, num);
        matrix_map actvec(avec, m, num);

        grad.noalias() = actvec * errvec.transpose();
    }

    // update word embedding gradient
    for (unsigned int i = 0; i < num; i++) {
        double* evec = get_error(0);
        double* gvec = get_gradient(layer_number - 1);
        unsigned int* context = input + i * order;
        matrix_map gmat(gvec, feature_size, (order - 1) * num);
        matrix_map emat(evec, feature_size, (order - 1) * num);

        for (unsigned int j = 0; j < order - 1; j++) {
            auto err = emat.col(i * (order - 1) + j);
            auto grad = gmat.col(i * (order - 1) + j);

            grad = err;
            queue[queue_size++] = context[j];
        }
    }

    delete[] label;

    return cost;
}

// training neural network, batch version
double trainer::train(double* x, unsigned int* y, unsigned int n)
{
    double cost;
    unsigned int in_size = layer_size[0];

    // copy input to input layer
    for (unsigned int i = 0; i < n; i++) {
        matrix_map in(x + i * in_size, in_size, 1);
        matrix_map neu(neuron, in_size + 1, n);
        auto act = neu.block(1, 0, in_size, n);

        act.col(i) = in;
    }

    cost = backpropagation(y, n);

    // calculate gradient
    for (unsigned int i = 0; i < layer_number - 1; i++) {
        double* avec = get_layer(i);
        double* g = get_gradient(i);
        double* err = get_error(i + 1);
        unsigned int inum = layer_size[i] + 1;
        unsigned int onum = layer_size[i + 1];
        matrix_map grad(g, inum, onum);
        matrix_map errvec(err, onum, n);
        matrix_map actvec(avec, inum, n);

        grad.noalias() = actvec * errvec.transpose();
    }

    return cost;
}

void trainer::initialize_parameter(const std::string& s, double l, double u)
{
    // initialize parameters
    if (s == "none")
        return;

    random(embedding, l, u, neural_model->get_embedding_size());

    if (s == "random") {
        random_initialize(l, u);
    } else {
        uniform_initialize(l, u);
    }
}

void trainer::set_learninig_rate(double a)
{
    control.learning_rate = a;
}

void trainer::set_batch_size(unsigned int n)
{
    control.batch_size = n;
}

void trainer::set_normalization_control(double a)
{
    control.norm_control = a;
}

void trainer::set_update_range(double l, double u)
{
    gradient_constraint.set_limit(l, u);
}

void trainer::set_parameter_range(double l, double u)
{
    parameter_constraint.set_limit(l, u);
}

void trainer::set_error_function(const std::string& name)
{
    if (error_function != nullptr)
        delete error_function;

    error_function = new error_evaluator;
    error_function->pointer = this;

    if (name == "log")
        error_function->method = &trainer::log_likelihood;
    else if (name == "entropy")
        error_function->method = &trainer::cross_entropy;
    else
        error_function->method = &trainer::self_normalize;
}

void trainer::forward(unsigned int num)
{
    // forward pass
    for (unsigned int i = 0; i < layer_number - 1; i++) {
        unsigned int m = layer_size[i] + 1;
        unsigned int n = layer_size[i + 1];
        double* in = get_layer(i);
        double* out = get_layer(i + 1);
        double* weight = get_weight(i);
        matrix_map xmat(in, m, num);
        matrix_map wmat(weight, m, n);
        matrix_map amat(out, n + 1, num);
        auto ymat = amat.block(1, 0, n, num);
        auto act_func = activation_function[i + 1];

        ymat.noalias() = wmat.transpose() * xmat;

        if (act_func != static_cast<activation_handler>(identity)) {
            auto lambda = [act_func](double v) { return act_func(v); };
            ymat.noalias() = ymat.unaryExpr(lambda);
        }
    }
}

void trainer::backward(unsigned int num)
{
    // backward pass
    for (unsigned int i = layer_number - 2; i >= 0; i--) {
        unsigned int m = layer_size[i];
        unsigned int n = layer_size[i + 1];
        double* neu = get_layer(i);
        double* cerror = get_error(i);
        double* perror = get_error(i + 1);
        double* weight = get_weight(i);
        matrix_map cerrmat(cerror, m, num);
        matrix_map perrmat(perror, n, num);
        auto wmat = matrix_map(weight, m + 1, n).block(1, 0, m, n);

        cerrmat.noalias() = wmat * perrmat;

        if (i == 0)
            break;

        auto der_func = derivative(activation_function[i]);

        if (der_func == identity_derivative)
            continue;

        auto actmat = matrix_map(neu, m + 1, num).block(1, 0, m, num);
        auto lambda = [der_func](double v) { return der_func(v); };
        cerrmat.noalias() = cerrmat.cwiseProduct(actmat.unaryExpr(lambda));
    }
}

void trainer::uniform_initialize(double l, double u)
{
    double val;
    unsigned int v;
    unsigned int n = layer_number;
    parameter* param = neural_model->get_parameter();

    param->get_parameter("output-size", v);
    val = -std::log(v);

    for (unsigned int i = 0; i < n - 1; i++) {
        unsigned int inum = layer_size[i] + 1;
        unsigned int onum = layer_size[i + 1];
        double* weight = get_weight(i);
        auto out_weight = matrix_map(weight, inum, onum);

        for (unsigned int j = 0; j < inum; j++) {
            for (unsigned int k = 0; k < onum; k++) {
                if (i == n - 2 && j == 1)
                    out_weight(j, k) = val;
                else
                    out_weight(j, k) = random(l, u);
            }
        }
    }
}

void trainer::random_initialize(double l, double u)
{
    unsigned int n = layer_number;

    for (unsigned int i = 0; i < n - 1; i++) {
        unsigned int inum = layer_size[i];
        unsigned int onum = layer_size[i + 1];
        unsigned int snum = (inum + 1) * onum;
        double factor = std::sqrt(6.0 / (inum + onum));
        double* q = get_weight(i);

        for (unsigned int j = 0; j < snum; j++) {
            q[j] = factor * random(l, u);
        }
    }
}

double trainer::cross_entropy(unsigned int* y, unsigned int num)
{
    double cost = 0.0;
    double n = layer_size[layer_number - 1];
    double* e = get_error(layer_number - 1);
    double* a = get_layer(layer_number - 1);
    matrix_map neu(a, n + 1, num);
    matrix_map err(e, n, num);
    auto act = neu.block(1, 0, n, num);
    omp_lock_t lock;

    omp_init_lock(&lock);

    #pragma omp parallel for
    for (unsigned int i = 0; i < num; i++) {
        auto avec = act.col(i);
        auto evec = err.col(i);
        unsigned int label = y[i];
        double error = 0.0;

        for (unsigned int j = 0; j < n; j++) {
            double v = avec(j);

            if (j == label) {
                error += -std::log(v);
                evec(j) = -1.0 / v;
            } else {
                error += -std::log(1 - v);
                evec(j) = 1.0 / (1 - v);
            }
        }

        omp_set_lock(&lock);
        cost += error;
        omp_unset_lock(&lock);
    }

    return cost;
}

double trainer::log_likelihood(unsigned int* y, unsigned int num)
{
    double cost = 0.0;
    double n = layer_size[layer_number - 1];
    double* e = get_error(layer_number - 1);
    double* a = get_layer(layer_number - 1);
    matrix_map neu(a, n + 1, num);
    matrix_map err(e, n, num);
    auto act = neu.block(1, 0, n, num);
    omp_lock_t lock;

    omp_init_lock(&lock);

    #pragma omp parallel for
    for (unsigned int i = 0; i < num; i++) {
        double sum;
        double max;
        unsigned int label = y[i];
        auto avec = act.col(i);
        auto evec = err.col(i);

        max = avec.maxCoeff();
        evec = avec.unaryExpr([max](double v) { return std::exp(v - max); });
        sum = evec.sum();
        evec = evec / sum;
        evec(label) -= 1;
        omp_set_lock(&lock);
        cost += -(avec(label) - std::log(sum) - max);
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);

    return cost;
}

double trainer::self_normalize(unsigned int* y, unsigned int num)
{
    double alpha;
    double cost = 0.0;
    double n = layer_size[layer_number - 1];
    double* e = get_error(layer_number - 1);
    double* a = get_layer(layer_number - 1);
    matrix_map neu(a, n + 1, num);
    matrix_map err(e, n, num);
    auto act = neu.block(1, 0, n, num);
    omp_lock_t lock;

    omp_init_lock(&lock);
    alpha = control.norm_control;

    #pragma omp parallel for
    for (unsigned int i = 0; i < num; i++) {
        double sum;
        double factor;
        unsigned int label = y[i];
        auto avec = act.col(i);
        auto evec = err.col(i);

        evec = avec.unaryExpr([](double v) { return std::exp(v); });
        sum = evec.sum();
        evec = evec / sum;
        factor = std::log(sum);
        evec *= 1 + 2 * alpha * factor;
        evec(label) -= 1;

        omp_set_lock(&lock);
        cost += -(avec(label) - factor - alpha * factor * factor);
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);

    return cost;
}

double trainer::backpropagation(unsigned int* label, unsigned int num)
{
    double cost;

    forward(num);

    // compute cost and initialize error

    cost = error_function->evaluate(label, num);

    auto der_func = derivative(activation_function[layer_number - 1]);

    if (der_func != identity_derivative) {
        unsigned int n = layer_number - 1;
        unsigned int onum = layer_size[n];
        matrix_map errmat(get_error(n), onum, num);
        matrix_map neumat(get_layer(n), onum + 1, num);
        auto actmat = neumat.block(1, 0, onum, num);

        errmat.noalias() = errmat.cwiseProduct(actmat.unaryExpr(der_func));
    }

    backward(num);

    return cost;
}


double trainer::error_evaluator::evaluate(unsigned int* y, unsigned int n)
{
    return (pointer->*method)(y, n);
}

} /* lm */
} /* infinity */
