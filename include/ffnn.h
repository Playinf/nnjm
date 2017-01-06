/*
 * ffnn.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __FFNN_H__
#define __FFNN_H__

#include <handler.h>

namespace infinity {
namespace lm {

class ffnn {
public:
    ffnn();
    ~ffnn();

    ffnn(const ffnn&) = delete;
    ffnn& operator=(const ffnn&) = delete;

    double* get_parameter() const;
    unsigned int get_layer_number() const;
    unsigned int get_parameter_number() const;
    unsigned int get_activation_number() const;
    unsigned int get_layer_size(unsigned int n) const;
    activation_handler get_activation_function(unsigned int i) const;

    double* compute(double* act);
    void compute(double* in, double* act);
    void compute(double* in, double** neuron);
    void initialize(unsigned int* size, unsigned int n);
    void set_activation_function(activation_handler f, unsigned int i);
private:
    double* parameter;
    unsigned int* layer_size;
    unsigned int layer_number;
    unsigned int parameter_number;
    unsigned int activation_number;
    activation_handler* activation_function;
};

} /* lm */
} /* infinity */

#endif /* __FFNN_H__ */
