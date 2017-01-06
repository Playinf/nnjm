/* main.cpp */
#include <map>
#include <ctime>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <exception>
#include <manager.h>
#include <utility.h>

typedef std::map<std::string, std::vector<std::string>> map_type;

static unsigned int find_param(const std::string& param, int argc, char** argv)
{
    const unsigned int no_param = static_cast<unsigned int>(-1);

    for (int i = 0; i < argc; i++) {
        if (std::string(argv[i]) == param) {
            if (i + 1 < argc) {
                return i + 1;
            } else {
                std::cerr << "require a parameter" << std::endl;
                std::exit(-1);
            }
        }
    }

    return no_param;
}

int main(int argc, char** argv)
{
    int mode;
    unsigned int index1 = find_param("--training-file", argc, argv);
    unsigned int index2 = find_param("--testing-file", argc, argv);
    unsigned int index3 = find_param("--model-file", argc, argv);
    unsigned int no_param = static_cast<unsigned int>(-1);
    infinity::lm::manager model_manager;

    if (index1 == no_param && index2 == no_param) {
        std::cerr << "error: training-file or testing-file required";
        std::cerr << std::endl;
        std::exit(-1);
    }

    if (index1 != no_param && index2 != no_param) {
        std::cerr << "error: both training file and testing file provided";
        std::cerr << std::endl;
        std::exit(-1);
    }

    if (index3 == no_param) {
        std::cerr << "error: model-file parameter required" << std::endl;
        std::exit(-1);
    }

    if (index1 != no_param)
        mode = 1;
    else
        mode = 2;

    infinity::lm::init_multithread();

    try {
        model_manager.manage(argv, argc);

        if (mode) {
            auto t1 = std::clock();
            model_manager.train();
            auto t2 = std::clock();
            std::cerr << "training complete, took ";
            std::cerr << (t2 - t1) / CLOCKS_PER_SEC << " seconds" << std::endl;
        } else {
            auto t1 = std::clock();
            model_manager.test();
            auto t2 = std::clock();
            std::cerr << "training complete, took ";
            std::cerr << (t2 - t1) / CLOCKS_PER_SEC << " seconds" << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        std::exit(-1);
    }
}
