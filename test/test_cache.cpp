#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cache.h>
#include <utility.h>

using namespace infinity::lm;

int main()
{
    cache model_cache;

    model_cache.resize(1000000);

    std::ifstream file;

    file.open("run1.out");

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

        if (model_cache.find(input, 15) == nullptr) {
            model_cache.update(input, 15, 0.0);
        }

        ++count;

        if (count % 1000 == 0) {
            std::cerr << count << std::endl;
        }
    }

    std::cerr << model_cache.get_hit_rate() << std::endl;
}
