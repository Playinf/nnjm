/*
 * nnjm.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __NNJM_H__
#define __NNJM_H__

#include <mutex>

namespace infinity {
namespace lm {

class cache;
class model;
class vocab;

class nnjm {
public:
    typedef void (*output_handler)(double*, unsigned int);

    nnjm();
    ~nnjm();

    nnjm(const nnjm&) = delete;
    nnjm& operator=(const nnjm&) = delete;

    double get_hit_rate() const;
    unsigned int get_order() const;
    vocab* get_source_vocab() const;
    vocab* get_target_vocab() const;
    vocab* get_output_vocab() const;
    unsigned int get_source_context() const;
    unsigned int get_target_context() const;

    void precompute();
    void load(const char* name);
    void set_cache_size(unsigned int n);
    double probability(unsigned int* input);
private:
    double* memory;
    double* embedding;
    std::mutex mutex;
    cache* model_cache;
    vocab* source_vocab;
    vocab* target_vocab;
    vocab* output_vocab;
    model* neural_model;
    unsigned int flag;
    unsigned int order;
    unsigned int input_number;
    unsigned int output_number;
    unsigned int feature_number;
    unsigned int source_context;
    unsigned int target_context;
    unsigned int activation_number;
    output_handler output_function;
};

} /* lm */
} /* infinity */

#endif /* __NNJM_H__ */
