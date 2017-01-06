#include <omp.h>
#include <iostream>

int main()
{
    int num = omp_get_max_threads();
    #pragma omp parallel for
    for (unsigned int i = 0; i < 1; i++) {
        std::cerr << i << std::endl;
    }

    std::cerr << num << std::endl;
    std::cerr << omp_get_num_threads() << std::endl;
}
