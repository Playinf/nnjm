#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nnltm.h>
#include <utility.h>

using namespace infinity::lm;

int main()
{
    nnltm tm_model;
    std::ifstream file;
    std::ofstream outfile;

    tm_model.load("model-iter-1.nnltm");

    tm_model.set_cache_size(1000000);
    std::cerr << "do premultiply";
    tm_model.precompute();
    std::cerr << "... complete" << std::endl;

    file.open("nnltm.test");
    outfile.open("run1.result");

    if (file.fail()) {
        std::cerr << "cannot open run1.result" << std::endl;
        return -1;
    }

    std::string line;
    unsigned int input[15];
    unsigned int count = 0;

    while (std::getline(file, line)) {
        std::vector<std::string> vec;

        string_split(line, " ", vec);

        if (vec.size() != 15) {
            std::cerr << count << std::endl;
            return -1;
        }

        for (unsigned int i = 0; i < vec.size(); i++) {
            input[i] = std::stoi(vec[i]);
        }

        double val = tm_model.ngram_prob(input);

        outfile << val << std::endl;

        ++count;

        if (count % 1000 == 0) {
            std::cerr << count << std::endl;
            std::cerr << tm_model.get_hit_rate() << std::endl;
            std::cerr << tm_model.get_precompute_hit_rate() << std::endl;
        }
    }
}
