/*
 * nnltm.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __NNLTM_H__
#define __NNLTM_H__

#include <mutex>

namespace infinity {
namespace lm {

class cache;
class model;
class vocab;

class nnltm {
public:
    typedef void (*output_handler)(double*, unsigned int);

    nnltm();
    ~nnltm();

    nnltm(const nnltm&) = delete;
    nnltm& operator=(const nnltm&) = delete;

    double get_hit_rate() const;
    vocab* get_input_vocab() const;
    vocab* get_output_vocab() const;
    unsigned int get_window_size() const;

    void precompute();
    void load(const char* name);
    void set_cache_size(unsigned int n);
    double probability(unsigned int* input);
private:
    double* memory;
    double* embedding;
    std::mutex mutex;
    cache* model_cache;
    vocab* input_vocab;
    vocab* output_vocab;
    model* neural_model;
    unsigned int flag;
    unsigned int window_size;
    unsigned int input_number;
    unsigned int output_number;
    unsigned int feature_number;
    unsigned int activation_number;
    output_handler output_function;
};

} /* lm */
} /* infinity */

#endif /* __NNLTM_H__ */
