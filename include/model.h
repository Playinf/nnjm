/*
 * model.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __MODEL_H__
#define __MODEL_H__

namespace infinity {
namespace lm {

class ffnn;
class vocab;
class parameter;

class model {
public:
    model();
    ~model();

    model(model&) = delete;
    model& operator=(const model&) = delete;

    ffnn* get_network() const;
    double* get_weight() const;
    double* get_embedding() const;
    vocab* get_source_vocab() const;
    vocab* get_target_vocab() const;
    vocab* get_output_vocab() const;
    parameter* get_parameter() const;
    unsigned int get_order() const;
    unsigned int get_window() const;
    unsigned int get_input_size() const;
    unsigned int get_output_size() const;
    unsigned int get_feature_size() const;
    unsigned int get_embedding_size() const;
    unsigned int get_source_context() const;
    unsigned int get_target_context() const;
    unsigned int get_activation_size() const;

    void initialize();
    double* compute(double* a);
private:
    ffnn* network;
    double* embedding;
    vocab* source_vocab;
    vocab* target_vocab;
    vocab* output_vocab;
    unsigned int order;
    unsigned int window;
    unsigned int input_size;
    unsigned int output_size;
    unsigned int feature_size;
    unsigned int source_context;
    unsigned int target_context;
    unsigned int embedding_size;
    unsigned int activation_size;
    parameter* model_parameter;
};

} /* lm */
} /* infinity */

#endif /* __MODEL_H__ */
