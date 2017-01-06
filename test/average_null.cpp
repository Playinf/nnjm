#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <exception>
#include <io.h>
#include <model.h>
#include <vocab.h>
#include <utility.h>

int main(int argc, char** argv)
{
    std::ifstream file;

    if (argc != 4) {
        std::cerr << "average train in-model out-model" << std::endl;
        return -1;
    }

    file.open(argv[1]);

    if (file.fail()) {
        std::cerr << "cannot open " << argv[1] << std::endl;
        return -1;
    }

    infinity::lm::model neural_model;

    try {
        infinity::lm::load_model(argv[2], &neural_model);
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    unsigned int fnum = neural_model.get_feature_size();
    unsigned int in_num = neural_model.get_input_size();
    std::string line;
    double* weight = new double[in_num];

    for (unsigned int i = 0; i < in_num; i++) {
        weight[i] = 0.0;
    }

    while (std::getline(file, line)) {
        std::vector<std::string> str_vec;
        infinity::lm::vocab* target_vocab = neural_model.get_target_vocab();

        infinity::lm::string_split(line, " ", str_vec);
        unsigned int size = str_vec.size();

        unsigned int index = target_vocab->get_id(str_vec[size - 2]);
        weight[index] += 1.0;
    }

    double* average = new double[fnum];
    double* embedding = neural_model.get_embedding();

    for (unsigned int i = 0; i < fnum; i++)
        average[i] = 0.0;

    for (unsigned int i = 0; i < fnum; i++) {
        for (unsigned int j = 0; j < in_num; j++) {
            double* feature = embedding + j * fnum;

            average[i] += feature[i] * weight[i];
        }
    }

    double sum = 0.0;

    for (unsigned int i = 0; i < in_num; i++)
        sum += weight[i];

    for (unsigned int i = 0; i < fnum; i++)
        average[i] /= sum;

    double* null = embedding + fnum;

    for (unsigned int i = 0; i < fnum; i++)
        null[i] = average[i];

    try {
        infinity::lm::save_model(argv[3], &neural_model);
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    delete[] weight;
    delete[] average;
}
