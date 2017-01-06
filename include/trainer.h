/*
 * trainer.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __TRAINER_H__
#define __TRAINER_H__

#include <string>
#include <handler.h>
#include <constraint.h>

namespace infinity {
namespace lm {

class model;

class trainer {
public:
    trainer();
    ~trainer();

    trainer(const trainer&) = delete;
    trainer& operator=(const trainer&) = delete;

    double* get_error(unsigned int i) const;
    double* get_layer(unsigned int i) const;
    double* get_weight(unsigned int i) const;
    double* get_gradient(unsigned int i) const;
    unsigned int get_size(unsigned int i) const;

    void initialize(ffnn* n);
    void initialize(model* m);

    void compute(double* x);
    void update_parameter();
    double train(unsigned int* x);
    double train(double* x, unsigned int y);
    double train(unsigned int* x, unsigned int n);
    double train(double* x, unsigned int* y, unsigned int n);
    void initialize_parameter(const std::string& s, double u, double l);

    void set_learninig_rate(double a);
    void set_batch_size(unsigned int n);
    void set_normalization_control(double a);
    void set_update_range(double l, double u);
    void set_parameter_range(double l, double u);
    void set_error_function(const std::string& s);
private:
    void forward(unsigned int num);
    void backward(unsigned int num);
    void random_initialize(double l, double u);
    void uniform_initialize(double l, double u);
    double cross_entropy(unsigned int* y, unsigned int n);
    double log_likelihood(unsigned int* y, unsigned int n);
    double self_normalize(unsigned int* y, unsigned int n);
    double backpropagation(unsigned int* input, unsigned int n);
private:
    struct control_type {
        double learning_rate;
        double weight_decay;
        double initial_momentum;
        double final_momentum;
        double norm_control;
        unsigned int batch_size;
        unsigned int momentum_control;
    };

    struct offset_type {
        unsigned int layer_offset;
        unsigned int weight_offset;
        unsigned int gradient_offset;
    };

    struct error_evaluator {
        typedef double (trainer::*function_type)(unsigned int*, unsigned int);
        double evaluate(unsigned int* y, unsigned int n);

        trainer* pointer;
        function_type method;
    };
private:
    ffnn* network;
    double* error;
    double* neuron;
    double* synapse;
    double* gradient;
    double* embedding;
    model* neural_model;
    offset_type* offset;
    control_type control;
    unsigned int* queue;
    unsigned int queue_size;
    unsigned int* layer_size;
    unsigned int layer_number;
    error_evaluator* error_function;
    activation_handler* activation_function;
    numeric_constraint gradient_constraint;
    numeric_constraint parameter_constraint;
};

} /* lm */
} /* infinity */

#endif /* __TRAINER_H__ */
