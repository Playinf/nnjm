/* manager.cpp */
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <io.h>
#include <misc.h>
#include <model.h>
#include <vocab.h>
#include <manager.h>
#include <mathlib.h>
#include <trainer.h>
#include <utility.h>
#include <parameter.h>

namespace infinity {
namespace lm {

static bool is_option(const char* str)
{
    if (str == nullptr)
        return false;

    std::string opt_str(str);
    unsigned int len = opt_str.size();

    if (len > 0 && opt_str[0] != '-')
        return false;

    if (len > 1) {
        unsigned int val;
        val = opt_str.substr(1, 1).find_first_not_of("0123456789");
        if (val == 0)
            return true;
    }

    return false;
}

static void set_parameter(char** args, unsigned int n, map_type& pmap)
{
    for (unsigned int i = 0; i < n; i++) {
        if (is_option(args[i])) {
            unsigned int start_pos = i + 1;
            unsigned int index = 0;
            std::string param(args[i] + 2);

            pmap[param];

            while (start_pos < n && !is_option(args[start_pos])) {
                if (pmap[param].size() > index)
                    pmap[param][index] = args[start_pos];
                else
                    pmap[param].push_back(args[start_pos]);
                index++;
                start_pos++;
            }
        }
    }
}

manager::manager()
{
    order = 0;
    feature_size = 0;
    activation_size = 0;
    source_vocab = nullptr;
    target_vocab = nullptr;
    output_vocab = nullptr;
    neural_model = nullptr;
    output_function = nullptr;

}

manager::~manager()
{
    if (neural_model != nullptr)
        delete neural_model;
}

void manager::train()
{
    double alpha;
    double norm_control;
    parameter* param;
    unsigned int max_epoch;
    unsigned int ini_epoch;
    unsigned int batch_size;
    trainer model_trainer;
    std::ifstream train_file;
    std::ifstream valid_file;
    model* m = neural_model;
    double prev_prob;
    double curr_prob;
    std::string init_method;
    std::string cost_func;
    std::vector<double> init_range;
    std::vector<double> update_range;
    std::vector<double> weight_range;

    param = m->get_parameter();
    param->get_parameter("epoch", ini_epoch);
    param->get_parameter("max-epoch", max_epoch);
    param->get_parameter("batch-size", batch_size);
    param->get_parameter("learning-rate", alpha);
    param->get_parameter("norm-control", norm_control);
    param->get_parameter("init-method", init_method);
    param->get_parameter("cost-function", cost_func);
    param->get_parameter("init-range", init_range);
    param->get_parameter("update-range", update_range);
    param->get_parameter("weight-range", weight_range);

    model_trainer.set_learninig_rate(alpha);
    model_trainer.set_batch_size(batch_size);
    model_trainer.set_normalization_control(norm_control);
    model_trainer.set_error_function(cost_func);
    model_trainer.set_update_range(update_range[0], update_range[1]);
    model_trainer.set_parameter_range(weight_range[0], weight_range[1]);
    model_trainer.initialize(m);

    if (init_method != "none") {
        double min = init_range[0];
        double max = init_range[1];
        model_trainer.initialize_parameter(init_method, min, max);
        param->set_parameter("init-method", "none");
    }

    prev_prob = -1000000.0;

    for (unsigned int iter = ini_epoch + 1; iter <= max_epoch; iter++) {
        double time;
        double start;
        double end;
        double cost = 0;
        double logp = 0;
        unsigned int count = 0;
        std::string line;

        train_file.open(training_filename);

        if (train_file.fail()) {
            std::cerr << "cannot open " << training_filename << std::endl;
            std::exit(-1);
        }

        unsigned int* input = new unsigned int[order * batch_size];

        time = std::clock();
        start = time;

        while (!train_file.eof()) {
            double j;
            unsigned int bnum = 0;

            for (unsigned int i = 0; i < batch_size; i++) {
                std::string line;
                std::vector<std::string> vec;

                if (!getline(train_file, line))
                    break;

                ++bnum;
                ++count;
                string_split(line, " ", vec);

                if (vec.size() != order) {
                    std::cerr << "error: not compatible with order";
                    std::cerr << std::endl;
                    std::cerr << line << std::endl;
                    std::exit(-1);
                }

                make_ngram(vec, input + i * order);
            }

            start = std::clock();
            j = model_trainer.train(input, bnum);
            model_trainer.update_parameter();
            end = std::clock();

            double result = bnum * CLOCKS_PER_SEC / (end - start);
            std::cerr << "\rwords/sec: " << result << std::flush;

            cost += j;
            logp += -std::log10(2) * j;
        }

        time = std::clock() - time;
        double sec = time / static_cast<double>(CLOCKS_PER_SEC);

        std::cerr << std::endl;
        std::cerr << "time: " << sec << "seconds" << std::endl;
        std::cerr << "iter " << iter << " cost: " << cost << std::endl;
        std::cerr << "average " << cost / count << std::endl;

        train_file.close();

        param->set_parameter("epoch", iter);
        std::cerr << "saving model... ";
        save_model(model_filename.c_str(), neural_model);
        std::cerr << "complete" << std::endl;
        delete[] input;

        valid_file.open(validation_filename);

        if (valid_file.fail())
            continue;

        valid_file.close();
        curr_prob = test_file(validation_filename.c_str());

        if (curr_prob < prev_prob)
            alpha = alpha * 0.5;
        prev_prob = curr_prob;
    }
}

void manager::test()
{
    double logp;

    logp = test_file(testing_filename.c_str());

    std::cerr << "testing result: " << logp << std::endl;
}

void manager::manage(char** args, unsigned int n)
{
    map_type pmap;
    std::string model_name;
    std::ifstream model_file;
    parameter* param;
    unsigned int ssize;
    unsigned int tsize;
    unsigned int osize;

    set_parameter(args, n, pmap);

    model_name = pmap["model-file"][0];

    neural_model = new model;
    param = neural_model->get_parameter();
    source_vocab = neural_model->get_source_vocab();
    target_vocab = neural_model->get_target_vocab();
    output_vocab = neural_model->get_output_vocab();
    model_file.open(model_name.c_str(), std::ios::binary);

    if (model_file.fail()) {
        std::string model;
        std::vector<std::string> invocab;
        std::string ovocab;
        std::cerr << "writing to " << model_name << std::endl;
        load_parameter(pmap, param);
        param->get_parameter("model", model);
        param->get_parameter("input-vocab", invocab);
        param->get_parameter("output-vocab", ovocab);

        if (model == "nnjm") {
            std::cerr << "reading source vocabulary from " << invocab[0];
            load_vocab(invocab[0].c_str(), source_vocab);
            std::cerr << std::endl;
            std::cerr << "reading target vocabulary from " << invocab[1];
            load_vocab(invocab[1].c_str(), target_vocab);
            std::cerr << std::endl;
            std::cerr << "reading output vocabulary from " << ovocab;
            load_vocab(ovocab.c_str(), output_vocab);
            std::cerr << std::endl;
        } else {
            std::cerr << "reading input vocabulary from " << invocab[0];
            load_vocab(invocab[0].c_str(), target_vocab);
            std::cerr << std::endl;
            std::cerr << "reading output vocabulary from " << ovocab;
            load_vocab(ovocab.c_str(), output_vocab);
            std::cerr << std::endl;
        }

        ssize = source_vocab->get_size();
        tsize = target_vocab->get_size();
        osize = output_vocab->get_size();

        param->get_parameter("input-size", input_size);
        param->get_parameter("output-size", output_size);

        if (ssize + tsize > input_size) {
            input_size = ssize + tsize;
            param->set_parameter("input-size", input_size);
            std::cerr << "change input-size to vocabulary size" << std::endl;
        }

        if (osize > output_size) {
            output_size = osize;
            param->set_parameter("output-size", output_size);
            std::cerr << "change output-size to vocabulary size" << std::endl;
        }

        check_parameter(param);
        neural_model->initialize();
    } else {
        model_file.close();
        std::cerr << "loading model from " << model_name << std::endl;
        load_model(model_name.c_str(), neural_model);
        overwrite_parameter(pmap, param);
        check_parameter(param);
    }

    unsigned int verbose = param->get_parameter("verbose", verbose);

    // output defined parameters
    if (verbose) {
        std::map<std::string, std::vector<std::string>> m;
        param->get_parameter(m);

        for (auto& val : m) {
            auto& name = val.first;
            auto& vec = val.second;

            std::cerr << name << ":";

            for (unsigned int i = 0; i < vec.size(); i++) {
                std::cerr << " " << vec[i];
            }
            std::cerr << std::endl;
        }
    }

    std::string out_func;

    order = neural_model->get_order();
    input_size = neural_model->get_input_size();
    output_size = neural_model->get_output_size();
    feature_size = neural_model->get_feature_size();
    activation_size = neural_model->get_activation_size();

    param->get_parameter("source-context", source_context);
    param->get_parameter("target-context", target_context);
    param->get_parameter("output-function", out_func);
    param->get_parameter("model-file", model_filename);
    param->get_parameter("training-file", training_filename);
    param->get_parameter("testing-file", testing_filename);
    param->get_parameter("validation-file", validation_filename);

    if (out_func == "softmax")
        output_function = softmax;
    else
        output_function = identity;
}

double manager::test_file(const char* name)
{
    std::ifstream test_file;
    model* m = neural_model;
    double* embedding = m->get_embedding();

    test_file.open(name);

    if (test_file.fail()) {
        std::cout << "cannot open " << name << std::endl;
        std::exit(-1);
    }

    std::string line;
    double logp = 0;
    unsigned int count = 0;
    unsigned int* input = new unsigned int[order];
    double* neuron = new double[activation_size];

    while (getline(test_file, line)) {
        std::vector<std::string> vec;
        ++count;

        string_split(line, " ", vec);

        if (vec.size() != order) {
            std::cerr << "error: not compatible with order";
            std::cerr << std::endl;
            std::cerr << line << std::endl;
            std::exit(-1);
        }

        make_ngram(vec, input);

        for (unsigned int i = 0; i < order - 1; i++) {
            double* act = neuron + i * feature_size;
            double* feature = embedding + input[i] * feature_size;

            for (unsigned int j = 0; j < feature_size; j++)
                act[j] = feature[j];
        }

        double* out = m->compute(neuron);
        output_function(out, output_size);
        logp += std::log10(out[input[order - 1]]);
    }

    test_file.close();

    delete[] input;
    delete[] neuron;

    return logp;
}

void manager::make_ngram(std::vector<std::string>& v, unsigned int* n)
{
    unsigned int index = 0;

    for (unsigned int i = 0; i < source_context; i++) {
        n[index] = source_vocab->get_id(v[index]);
        index++;
    }

    for (unsigned int i = 0; i < target_context; i++) {
        n[index] = target_vocab->get_id(v[index]);
        index++;
    }

    n[index] = output_vocab->get_id(v[index]);
}

} /* lm */
} /* infinity */
