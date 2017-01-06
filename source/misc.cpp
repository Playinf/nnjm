/* misc.cpp */
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <misc.h>
#include <parameter.h>

namespace infinity {
namespace lm {

static unsigned int get_limit(int type)
{
    if (type & parameter::type::vector) {
        return 100;
    } else {
        return 1;
    }
}

static void add_parameter(const std::string& name,
    std::vector<std::string>& val, parameter* param)
{
    unsigned int pnum = val.size();
    bool found = param->has_parameter(name);

    if (!found) {
        std::string msg = "unknow parameter " + name;
        throw std::runtime_error(msg);
    }

    if (pnum >= 1) {
        int type;
        unsigned int lim;

        param->get_type(name, type);
        lim = get_limit(type);

        if (pnum > lim) {
            std::string msg = "parameter " + name + " has too many values";
            throw std::runtime_error(msg);
        }

        if (type & parameter::type::real) {
            std::vector<double> vec;

            for (unsigned int i = 0; i < pnum; i++) {
                double v = std::stod(val[i]);
                vec.push_back(v);
            }

            param->set_parameter(name, vec);
        } else if (type & parameter::type::string) {
            param->set_parameter(name, val);

        } else if (type & parameter::type::integer) {
            std::vector<unsigned int> vec;

            for (unsigned int i = 0; i < pnum; i++) {
                unsigned int v = std::stoi(val[i]);
                vec.push_back(v);
            }

            param->set_parameter(name, vec);
        }
    }
}

void check_parameter(parameter* param)
{
    std::string model;

    param->get_parameter("model", model);

    if (model != "nnlm" && model != "nnjm" && model != "nnltm") {
        std::string msg = "model must be nnlm or nnjm or nnltm, not " + model;
        throw std::runtime_error(msg);
    }

    unsigned int order;
    unsigned int win_size = 0;
    unsigned int src_ctx = 0;
    unsigned int tgt_ctx = 0;

    param->get_parameter("order", order);
    param->get_parameter("window", win_size);
    param->get_parameter("source-context", src_ctx);
    param->get_parameter("target-context", tgt_ctx);

    if (model == "nnjm") {
        if (src_ctx == 0) {
            std::string msg = "source-context cannot be 0";
            throw std::runtime_error(msg);
        }

        if (tgt_ctx == 0) {
            std::string msg = "target-context cannot be 0";
            throw std::runtime_error(msg);
        }

        if (src_ctx + tgt_ctx + 1 != order) {
            std::cerr << "warning: ";
            std::cerr << "source-context + target-context + 1 != order, ";
            std::cerr << "using order = source-context + target-context + 1";
            std::cerr << std::endl;
            param->set_parameter("order", src_ctx + tgt_ctx + 1);
        }
    } else if (model == "nnlm") {
        if (src_ctx != 0) {
            std::string msg = "source-context must be 0";
            throw std::runtime_error(msg);
        }

        if (order == 0) {
            std::string msg = "order cannot be 0";
            throw std::runtime_error(msg);
        }

        if (tgt_ctx && tgt_ctx != order - 1) {
            std::string msg = "order != target-context + 1";
            throw std::runtime_error(msg);
        }

        if (tgt_ctx == 0) {
            param->set_parameter("target-context", order - 1);
        }
    } else {
        // nnltm
        if (win_size == 0) {
            std::string msg = "window cannot be 0";
            throw std::runtime_error(msg);
        }

        param->set_parameter("window", win_size);
    }

    std::string network;

    param->get_parameter("network", network);

    if (network != "ffnn") {
        std::string msg = network + " not supported";
        throw std::runtime_error(msg);
    }

    unsigned int feature_number;

    param->get_parameter("feature-number", feature_number);

    if (feature_number == 0) {
        std::string msg = "feature-number cannot be 0";
        throw std::runtime_error(msg);
    }

    unsigned int input_size;

    param->get_parameter("input-size", input_size);

    if (input_size == 0) {
        std::string msg = "input-size cannot be 0";
        throw std::runtime_error(msg);
    }

    unsigned int output_size;

    param->get_parameter("output-size", output_size);

    if (output_size == 0) {
        std::string msg = "output-size cannot be 0";
        throw std::runtime_error(msg);
    }

    std::string output_function;

    param->get_parameter("output-function", output_function);

    if (output_function != "softmax" && output_function != "identity") {
        std::string msg = output_function + " not supported";
        throw std::runtime_error(msg);
    }

    // training options
    double learning_rate;

    param->get_parameter("learning-rate", learning_rate);

    if (learning_rate <= 0) {
        std::string msg = "learning-rate <= 0";
        throw std::runtime_error(msg);
    }

    double weight_decay;

    param->get_parameter("weight-decay", weight_decay);

    if (learning_rate < 0) {
        std::string msg = "weight-decay < 0";
        throw std::runtime_error(msg);
    }

    double norm_contrl;

    param->get_parameter("norm-control", norm_contrl);

    if (norm_contrl < 0) {
        std::string msg = "norm-control < 0";
        throw std::runtime_error(msg);
    }

    unsigned int max_epoch;

    param->get_parameter("max-epoch", max_epoch);

    if (max_epoch == 0) {
        std::string msg = "max-epoch must bigger than 0";
        throw std::runtime_error(msg);
    }

    unsigned int epoch;

    param->get_parameter("epoch", epoch);

    if (epoch > max_epoch) {
        std::string msg = "epoch > max-epoch";
        throw std::runtime_error(msg);
    }

    unsigned int batch_size;

    param->get_parameter("batch-size", batch_size);

    if (batch_size == 0) {
        std::string msg = "batch-size cannot be 0";
        throw std::runtime_error(msg);
    } else if (batch_size > 10000) {
        std::string msg = "batch-size too large, cannot exceed 10,000";
        throw std::runtime_error(msg);
    }

    std::string cost_function;

    param->get_parameter("cost-function", cost_function);

    if (cost_function != "log" && cost_function != "entropy"
        && cost_function != "selfnorm") {
        std::string msg = cost_function + " not supported";
        throw std::runtime_error(msg);
    }

    std::string init_method;

    param->get_parameter("init-method", init_method);

    if (init_method != "random" && init_method != "uniform" &&
        init_method != "none") {
        std::string msg = init_method + " not supported";
        throw std::runtime_error(msg);
    }

    std::vector<unsigned int> hidden_size;
    unsigned int layer_number;

    param->get_parameter("hidden-size", hidden_size);

    layer_number = hidden_size.size() +2;
    param->set_parameter("layer-number", layer_number);

    std::vector<std::string> act_func;

    param->get_parameter("activation-function", act_func);

    if (act_func.size() > layer_number) {
        std::string msg = "too many activation-function specified";
        throw std::runtime_error(msg);
    } else  {
        unsigned int fnum = act_func.size();

        act_func.resize(layer_number);

        if (fnum < layer_number) {
            for (unsigned int i = 0; i < fnum; i++) {
                act_func[i + 1] = act_func[i];
            }
        }

        act_func[0] = "identity";

        for (unsigned int i = 1; i < layer_number; i++) {
            auto& func = act_func[i];

            if (func == "") {
                if (i == layer_number - 1)
                    func = "identity";
                else
                    func = "tanh";
            }

            if (func != "tanh" && func != "sigmoid" && func != "identity") {
                std::string msg = func + " not supported";
                throw std::runtime_error(msg);
            }
        }

        param->set_parameter("activation-function", act_func);
    }

    std::vector<double> momentum;

    param->get_parameter("momentum", momentum);

    if (momentum.size() == 1) {
        double val = momentum[0];

        if (val < 0.0) {
            std::string msg = "momentum cannot < 0";
            throw std::runtime_error(msg);
        }

        momentum.push_back(val);
        param->set_parameter("momentum", momentum);
    } else {
        double val1;
        double val2;

        if (momentum.size() != 2) {
            std::string msg = "too many momentum value specified";
            throw std::runtime_error(msg);
        }

        val1 = momentum[0];
        val2 = momentum[1];

        if (val1 < 0.0 || val2 < 0.0) {
            std::string msg = "momentum cannot < 0";
            throw std::runtime_error(msg);
        }
    }

    std::vector<double> init_range;

    param->get_parameter("init-range", init_range);

    if (init_range.size() != 2) {
        std::string msg = "init-range must provide 2 values";
        throw std::runtime_error(msg);
    }

    std::vector<double> update_range;

    param->get_parameter("update-range", update_range);

    if (update_range.size() != 2) {
        std::string msg = "update-range must provide 2 values";
        throw std::runtime_error(msg);
    }

    std::vector<double> weight_range;

    param->get_parameter("weight-range", weight_range);

    if (weight_range.size() != 2) {
        std::string msg = "weight-range must provide 2 values";
        throw std::runtime_error(msg);
    }
}

void load_parameter(map_type& setting, parameter* param)
{
    for (auto& val : setting) {
        auto& name = val.first;
        auto& vec = val.second;

        add_parameter(name, vec, param);
    }
}

void overwrite_parameter(map_type& setting, parameter* param)
{
    for (auto& val : setting) {
        auto& name = val.first;
        auto& value = val.second;
        int type;
        int flag;
        double dval;
        unsigned int ival;
        std::vector<double> dvec;
        std::vector<unsigned int> ivec;

        param->get_flag(name, flag);
        param->get_type(name, type);

        if (!flag) {
            std::cerr << "warning: new " << name << " ignored";
            std::cerr << std::endl;
            continue;
        }

        if (type & parameter::type::vector) {
            if (type & parameter::type::real) {
                for (unsigned int i = 0; i < value.size(); i++) {
                    double v = std::stod(value[i]);
                    dvec.push_back(v);
                }
                param->set_parameter(name, dvec);
            }

            if (type & parameter::type::string) {
                param->set_parameter(name, value);
            }

            if (type & parameter::type::integer) {
                for (unsigned int i = 0; i < value.size(); i++) {
                    unsigned int v = std::stoi(value[i]);
                    ivec.push_back(v);
                }
                param->set_parameter(name, ivec);
            }
        } else {
            if (type & parameter::type::real) {
                dval = std::stod(value[0]);
                param->set_parameter(name, dval);
            }

            if (type & parameter::type::string) {
                param->set_parameter(name, value[0]);
            }

            if (type & parameter::type::integer) {
                ival = std::stoi(value[0]);
                param->set_parameter(name, ival);
            }
        }
    }
}

} /* lm */
} /* infinity */
