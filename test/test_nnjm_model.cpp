#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nnjm.h>
#include <utility.h>

using namespace infinity::lm;

int main()
{
    nnjm jm_model;
    std::ifstream file;
    std::ofstream outfile;

    jm_model.load("output.nnjm");

    jm_model.set_cache_size(1000000);
    std::cerr << "do premultiply";
    jm_model.precompute();
    std::cerr << "... complete" << std::endl;

    file.open("call.txt");
    outfile.open("run1.result2");

    if (file.fail()) {
        std::cerr << "cannot open run1.out" << std::endl;
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

        double val = jm_model.ngram_prob(input);

        outfile << val << std::endl;

        ++count;

        if (count % 1000 == 0) {
            std::cerr << count << std::endl;
            std::cerr << jm_model.get_hit_rate() << std::endl;
            //std::cerr << jm_model.get_precompute_hit_rate() << std::endl;
        }
    }
}
