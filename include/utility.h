/*
 * utility.h
 *
 * author: Playinf
 * email: playinf@stu.edu.cn
 *
 */
#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <string>
#include <vector>

namespace infinity {
namespace lm {

void init_multithread();
void set_thread_number(unsigned int n);
double random(double min, double max);
void random(double* v, double min, double max, unsigned int n);

std::size_t hash_combine(std::size_t v1, std::size_t v2);
void string_trim(const std::string& str, std::string& ret);
void string_split(const std::string& s, const std::string& sep,
    std::vector<std::string>& vec);

} /* lm */
} /* infinity */

#endif /* __UTILITY_H__ */
