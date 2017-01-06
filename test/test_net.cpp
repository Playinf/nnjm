/* test_net.cpp */
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ffnn.h>
#include <mathlib.h>
#include <trainer.h>
#include <utility.h>

typedef double (*cost_handler)(double*, unsigned int, unsigned int);
typedef void (*out_handler)(double*, unsigned int);

int load_data(const char* name, double* buffer)
{
    std::ifstream file;
    std::string line;
    unsigned int index = 0;

    file.open(name);

    if (file.fail()) {
        std::cout << "cannot open " << name << std::endl;
        return -1;
    }

    while (std::getline(file, line)) {
        std::vector<std::string> vec;

        if (line[0] == '#')
            continue;

        string_split(line, " ", vec);

        if (vec.size() == 0)
            continue;

        for (unsigned int i = 0; i < vec.size(); i++)
            buffer[index++] = std::atof(vec[i].c_str());
    }

    file.close();

    return 0;
}


void copy_array(double* d, double* s, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

void initialize_parameter(ffnn& net)
{
    double* theta1;
    double* theta2;

    theta1 = new double[25 * 401];
    theta2 = new double[10 * 26];

    load_data("test/theta1.txt", theta1);
    load_data("test/theta2.txt", theta2);

    // initialize parameter
    double* param = net.get_parameter();
    unsigned int index = 0;

    for (unsigned int i = 0; i < 401; i++) {
        for (unsigned int j = 0; j < 25; j++) {
            param[index++] = *(theta1 + j * 401 + i);
        }
    }

    for (unsigned int i = 0; i < 26; i++) {
        for (unsigned int j = 0; j < 10; j++) {
            param[index++] = *(theta2 + j * 26 + i);
        }
    }

    index = 0;

    for (unsigned int i = 0; i < 25 * 401; i++) {
        param[index++] = theta1[i];
    }

    for (unsigned int i = 0; i < 26 * 10; i++) {
        param[index++] = theta2[i];
    }

    delete[] theta1;
    delete[] theta2;
}

void compute(trainer& t, double* data, double* output, unsigned int num,
             out_handler func)
{
    unsigned int inum = t.get_size(0);
    unsigned int onum = t.get_size(2);

    for (unsigned int i = 0; i < num; i++) {
        double* input = data + i * (inum + 1);
        double* out = output + i * onum;

        t.compute(input);

        double* act = t.get_layer(2) + 1;

        copy_array(out, act, onum);
        func(out, onum);
    }
}

void compute_gradient(trainer& t, ffnn& n, double* grad, double* data,
                      out_handler func1, cost_handler func2)
{
    double epsilon = 1e-6;
    unsigned int pnum = n.get_parameter_number();
    unsigned int inum = n.get_layer_size(0);
    double* param = n.get_parameter();

    for (unsigned int i = 0; i < pnum; i++) {
        unsigned int label = (unsigned int) data[inum] - 1;
        double val = param[i];
        param[i] = val + epsilon;
        t.compute(data);
        func1(t.get_layer(2) + 1, 10);
        double c1 = func2(t.get_layer(2) + 1, label, 10);
        param[i] = val - epsilon;
        t.compute(data);
        func1(t.get_layer(2) + 1, 10);
        double c2 = func2(t.get_layer(2) + 1, label, 10);
        param[i] = val;
        grad[i] = (c1 - c2) / (2 * epsilon);
    }
}

void display_gradient(trainer& t, double* data, double* ngrad)
{
    double diff = 0;
    double* grad[2];
    unsigned int index1 = 0;
    unsigned int index2 = 0;

    grad[0] = t.get_gradient(0);
    grad[1] = t.get_gradient(1);

    for (unsigned int i = 0; i < 401; i++) {
        for (unsigned int j = 0; j < 25; j++) {
            diff += std::abs(grad[0][index1] - ngrad[index2]);
            std::cout << grad[0][index1++] << " " << ngrad[index2++];
            std::cout << std::endl;
        }
    }

    index1 = 0;
    for (unsigned int i = 0; i < 26; i++) {
        for (unsigned int j = 0; j < 10; j++) {
            diff += std::abs(grad[1][index1] - ngrad[index2]);
            std::cout << grad[1][index1++] << " " << ngrad[index2++];
            std::cout << std::endl;
        }
    }

    std::cout << "diff: " << diff << std::endl;
}

void test_entropy()
{
    double* data;
    double* output;
    ffnn net;
    trainer learner;
    unsigned int size[] = { 400, 25, 10 };
    unsigned int num = 1;

    data = new double[5000 * 401];
    output = new double[5000 * 10];

    load_data("test/data-small.txt", data);

    net.initialize(size, 3);
    initialize_parameter(net);
    net.set_activation_function(sigmoid, 1);
    net.set_activation_function(sigmoid, 2);
    learner.initialize(&net);
    learner.set_error_function("entropy");

    compute(learner, data, output, num, identity);

    // compute cost
    double j1 = 0;
    double j2 = 0;

    for (unsigned int i = 0; i < num; i++) {
        double* input = data + i * 401;
        double* out = output + i * 10;
        unsigned int label = (unsigned int) input[400] - 1;

        j1 += entropy_cost(out, label, 10);
    }

    j1 /= num;
    j2 = learner.train(data, (unsigned int) data[400] - 1);

    std::cout << "j1: " << j1 << std::endl;
    std::cout << "j2: " << j2 << std::endl;
    std::system("pause");

    unsigned int pnum = net.get_parameter_number();
    double* ngrad = new double[pnum];
    compute_gradient(learner, net, ngrad, data, identity, entropy_cost);
    display_gradient(learner, data, ngrad);

    delete[] data;
    delete[] output;
}

void test_log()
{
    double* data;
    double* output;
    ffnn net;
    trainer learner;
    unsigned int size[] = { 400, 25, 10 };
    unsigned int num = 1;

    data = new double[5000 * 401];
    output = new double[5000 * 10];

    load_data("test/data-small.txt", data);

    net.initialize(size, 3);
    initialize_parameter(net);
    net.set_activation_function(sigmoid, 1);
    net.set_activation_function(sigmoid, 2);
    learner.initialize(&net);
    learner.set_error_function("log");

    compute(learner, data, output, num, softmax);

    // compute cost
    double j1 = 0;
    double j2 = 0;

    for (unsigned int i = 0; i < num; i++) {
        double* input = data + i * 401;
        double* out = output + i * 10;
        unsigned int label = (unsigned int) input[400] - 1;

        j1 += log_cost(out, label, 10);
    }

    j1 /= num;
    j2 = learner.train(data, (unsigned int) data[400] - 1);

    std::cout << "j1: " << j1 << std::endl;
    std::cout << "j2: " << j2 << std::endl;
    std::system("pause");

    unsigned int pnum = net.get_parameter_number();
    double* ngrad = new double[pnum];
    compute_gradient(learner, net, ngrad, data, softmax, log_cost);
    display_gradient(learner, data, ngrad);

    delete[] data;
    delete[] output;
}

void test_nce()
{
    double* data;
    double* output;
    ffnn net;
    trainer learner;
    unsigned int size[] = { 400, 25, 10 };
    unsigned int num = 1;

    data = new double[5000 * 401];
    output = new double[5000 * 10];

    load_data("test/data-small.txt", data);

    net.initialize(size, 3);
    initialize_parameter(net);
    net.set_activation_function(sigmoid, 1);
    net.set_activation_function(sigmoid, 2);
    learner.initialize(&net);
    learner.set_error_function("nce");
    learner.set_normalization_control(1.0);

    compute(learner, data, output, num, identity);

    // compute cost
    double j1 = 0;
    double j2 = 0;

    for (unsigned int i = 0; i < num; i++) {
        double* input = data + i * 401;
        double* out = output + i * 10;
        unsigned int label = (unsigned int) input[400] - 1;

        j1 += nce_cost(out, label, 10);
    }

    j1 /= num;
    j2 = learner.train(data, (unsigned int) data[400] - 1);

    std::cout << "j1: " << j1 << std::endl;
    std::cout << "j2: " << j2 << std::endl;
    std::system("pause");

    unsigned int pnum = net.get_parameter_number();
    double* ngrad = new double[pnum];
    compute_gradient(learner, net, ngrad, data, identity, nce_cost);
    display_gradient(learner, data, ngrad);

    delete[] data;
    delete[] output;
}

void test_batch()
{
    double* in;
    double* data;
    double* output;
    ffnn net;
    trainer learner1;
    trainer learner2;
    unsigned int size[] = { 400, 25, 10 };
    unsigned int num = 1;
    unsigned int* label;

    data = new double[5000 * 401];
    output = new double[5000 * 10];

    load_data("test/data-small.txt", data);

    net.initialize(size, 3);
    initialize_parameter(net);
    net.set_activation_function(sigmoid, 1);
    net.set_activation_function(sigmoid, 2);
    learner1.initialize(&net);
    learner1.set_error_function("entropy");
    learner2.set_batch_size(10);
    learner2.initialize(&net);
    learner2.set_error_function("entropy");

    compute(learner1, data, output, num, identity);

    // compute cost
    double j1 = 0;
    double j2 = 0;

    for (unsigned int i = 0; i < num; i++) {
        double* input = data + i * 401;
        double* out = output + i * 10;
        unsigned int label = (unsigned int) input[400] - 1;

        j1 += entropy_cost(out, label, 10);
    }

    j1 /= num;

    label = new unsigned int[10];
    in = new double[400 * 10];

    for (unsigned int i = 0; i < 10; i++) {
        double* src = data;
        double* dst = in + i * 400;

        copy_array(dst, src, 400);
        label[i] = (unsigned int) data[400] - 1;
    }

    j2 = learner2.train(in, label, 10);
    j2 /= 10.0;

    std::cout << "j1: " << j1 << std::endl;
    std::cout << "j2: " << j2 << std::endl;
    std::system("pause");

    unsigned int pnum = net.get_parameter_number();
    double* ngrad = new double[pnum];
    compute_gradient(learner1, net, ngrad, data, identity, entropy_cost);

    for (unsigned int i = 0; i < pnum; i++)
        ngrad[i] *= 10.0;

    display_gradient(learner2, data, ngrad);

    delete[] data;
    delete[] output;
    delete[] in;
    delete[] label;
}

int main()
{
    test_nce();
}
