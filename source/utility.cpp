/* utility.cpp */
#include <cmath>
#include <cstdlib>
#include <mathlib.h>
#include <utility.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num(x) 0
#define omp_set_num_threads(n)
#define omp_get_num_threads() 1
#define omp_get_max_threads() 1
#endif

namespace infinity {
namespace lm {

void init_multithread()
{
    unsigned int n = omp_get_max_threads();
    Eigen::initParallel();
    Eigen::setNbThreads(n);
}

void set_thread_number(unsigned int n)
{
    Eigen::setNbThreads(n);
}

double random(double min, double max)
{
    unsigned int num = std::rand();
    double frac = (double) num / RAND_MAX;

    return min + (max - min) * frac;
}

void random(double* v, double min, double max, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
        v[i] = random(min, max);
}

std::size_t hash_combine(std::size_t v1, std::size_t v2)
{
    std::size_t val = v1;

    val ^= v2 + 0x9e3779b9 + (v1 << 6) + (v1 >> 2);

    return val;
}

void string_trim(const std::string& str, std::string& ret)
{
    std::string::size_type pos1 = str.find_first_not_of(' ');
    std::string::size_type pos2 = str.find_last_not_of(' ');
    std::string::size_type npos = std::string::npos;

    if (pos1 == npos || pos2 == npos)
        ret = str;
    else
        ret = str.substr(pos1, pos2 - pos1 + 1);
}

void string_split(const std::string& s, const std::string& sep,
    std::vector<std::string>& vec)
{
    std::string::size_type len = s.length();
    std::string::size_type sep_len = sep.length();
    std::string::size_type start = 0;
    std::string::size_type pos = s.find(sep);
    std::string::size_type npos = s.npos;

    while (pos != npos) {
        std::string::size_type end = pos - start;

        if (end)
            vec.push_back(s.substr(start, end));

        start = pos + sep_len;
        pos = s.find(sep, start);
    }

    if (len > start)
        vec.push_back(s.substr(start, len));
}

} /* lm */
} /* infinity */
