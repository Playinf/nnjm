/*
 * nnlm.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __NNLM_H__
#define __NNLM_H__

#include <mutex>

namespace infinity {
namespace lm {

class cache;
class model;
class vocab;

class nnlm {
public:
    typedef void (*output_handler)(double*, unsigned int);

    nnlm();
    ~nnlm();

    nnlm(const nnlm&) = delete;
    nnlm& operator=(const nnlm&) = delete;

    double get_hit_rate() const;
    unsigned int get_order() const;
    vocab* get_input_vocab() const;
    vocab* get_output_vocab() const;

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
    unsigned int order;
    unsigned int input_number;
    unsigned int output_number;
    unsigned int feature_number;
    unsigned int activation_number;
    output_handler output_function;
};

} /* lm */
} /* infinity */

#endif /* __NNLM_H__ */
