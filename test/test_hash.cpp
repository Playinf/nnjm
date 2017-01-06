#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nnjm.h>
#include <utility.h>
#include <boost/functional/hash.hpp>

using namespace infinity::lm;

int main()
{
    std::ifstream file;
    std::ofstream hashfile;

    file.open("nplm-call.txt");
    hashfile.open("hashval.txt");

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

        std::size_t val = 0;

        //for (unsigned int i = 0; i < 15; i++) {
            //val = hash_combine(val, input[i]);
        //}

        //std::size_t hashval = 0;

        //for (unsigned int i = 0; i <  15; i++) {
            //boost::hash_combine(hashval, input[i]);
        //}

        //hashfile << val << std::endl;

        ++count;

        if (count % 1000 == 0) {
            std::cerr << count << std::endl;
            std::cerr << jm_model.get_hit_rate() << std::endl;
        }
    }
}
