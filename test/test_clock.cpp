#include <chrono>
#include <iostream>
#include <mathlib.h>

int main()
{
    std::chrono::steady_clock clk;
    unsigned int gigabytes = 1024 * 1024 * 1024;
    double* large_array = new double[gigabytes];
    unsigned int count = 10;

    auto t1 = clk.now();

    // copy about 8G memory, roughly 1,000,000 words with 10,000 features
    for (unsigned int i = 0; i < count; i++) {
        matrix_map mat(large_array, gigabytes, 1);

        mat = mat.unaryExpr([](double v) { return 1; });
    }

    // suppose we have 1,000,000 training sentence
    // each sentence have 1,000 words, we train 100 epoch
    // then we need 1,000,000 * 1,000 * 100 iteration

    auto t2 = clk.now();

    auto dms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << dms.count() << std::endl;

    delete[] large_array;
}
