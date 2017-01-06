#include <iostream>
#include <ffnn.h>
#include <library.h>

int main()
{
    ffnn net;
    unsigned int size[] = { 2, 2, 2 };
    double grad[9] = { 0.0 };

    net.initialize(size, 3);

    double w0[] = {
        0.0, 0.0,
        0.62, 0.42,
        0.55, -0.17,
    };

    double w1[] = {
        0.0,
        0.35,
        0.81,
    };

    net.set_parameter(w0, 0);
    net.set_parameter(w1, 1);

    double in1[] = { 0.0, 1.0 };
    double in2[] = { 1.0, 1.0 };
    int o1 = 0;
    int o2 = 1;

    ffnn::option opt;
    opt.alpha = 0.0;
    opt.beta = 0.0;

    net.train(in1, o2, opt);

}
