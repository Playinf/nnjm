/* model.cpp */
#include <cmath>
#include <fstream>
#include <iostream>
#include <exception>
#include <io.h>
#include <ffnn.h>
#include <model.h>
#include <vocab.h>
#include <mathlib.h>
#include <parameter.h>

namespace infinity {
namespace lm {

model::model()
{
    embedding = nullptr;
    network = new ffnn;
    source_vocab = new vocab;
    target_vocab = new vocab;
    output_vocab = new vocab;
    model_parameter = new parameter;
}

model::~model()
{
    if (network != nullptr)
        delete network;

    if (embedding != nullptr)
        delete[] embedding;

    if (source_vocab != nullptr)
        delete source_vocab;

    if (target_vocab != nullptr)
        delete target_vocab;

    if (output_vocab != nullptr)
        delete output_vocab;

    if (model_parameter != nullptr)
        delete model_parameter;
}

ffnn* model::get_network() const
{
    return network;
}

double* model::get_weight() const
{
    return network->get_parameter();
}

double* model::get_embedding() const
{
    return embedding;
}

vocab* model::get_source_vocab() const
{
    return source_vocab;
}

vocab* model::get_target_vocab() const
{
    return target_vocab;
}

vocab* model::get_output_vocab() const
{
    return output_vocab;
}

parameter* model::get_parameter() const
{
    return model_parameter;
}

unsigned int model::get_order() const
{
    return order;
}

unsigned int model::get_window() const
{
    return window;
}

unsigned int model::get_input_size() const
{
    return input_size;
}

unsigned int model::get_output_size() const
{
    return output_size;
}

unsigned int model::get_feature_size() const
{
    return feature_size;
}

unsigned int model::get_embedding_size() const
{
    return embedding_size;
}

unsigned int model::get_source_context() const
{
    return source_context;
}

unsigned int model::get_target_context() const
{
    return target_context;
}

unsigned int model::get_activation_size() const
{
    return activation_size;
}

void model::initialize()
{
    unsigned int layer_number;
    parameter* param = model_parameter;
    std::vector<std::string> act_func;
    std::vector<unsigned int> hidden_size;

    param->get_parameter("order", order);
    param->get_parameter("window", window);
    param->get_parameter("input-size", input_size);
    param->get_parameter("hidden-size", hidden_size);
    param->get_parameter("output-size", output_size);
    param->get_parameter("layer-number", layer_number);
    param->get_parameter("feature-number", feature_size);
    param->get_parameter("source-context", source_context);
    param->get_parameter("target-context", target_context);
    param->get_parameter("activation-function", act_func);

    embedding_size = feature_size * input_size;
    embedding = new double[embedding_size];

    unsigned int* size = new unsigned int[layer_number];
    ffnn* net = network;

    size[0] = feature_size * (order - 1);

    for (unsigned int i = 0; i < hidden_size.size(); i++) {
        size[i + 1] = hidden_size[i];
    }

    size[layer_number - 1] = output_size;

    net->initialize(size, layer_number);

    for (unsigned int i = 0; i < layer_number; i++) {
        std::string& func = act_func[i];
        activation_handler act_handler;

        if (func == "tanh") {
            act_handler = std::tanh;
        } else if (func == "sigmoid") {
            act_handler = sigmoid;
        } else {
            act_handler = identity;
        }

        net->set_activation_function(act_handler, i);
    }

    activation_size = net->get_activation_number();

    delete[] size;
}

double* model::compute(double* a)
{
    return network->compute(a);
}

} /* lm */
} /* infinity */
