#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ffnn.h>
#include <mathlib.h>
#include <trainer.h>
#include <utility.h>

#include "mnist.h"

void print_digit(unsigned char* img)
{
    unsigned int row = 28;
    unsigned int col = 28;
    unsigned int fnum = row * col;

    for (unsigned int i = 0; i < fnum; i++) {
        unsigned int val;
        val = img[i] * 10.0 / 255.1;

        std::cout << val << " ";

        if ((i + 1) % col == 0)
            std::cout << std::endl;
    }
}

void save_image(const char* name, unsigned char** img, unsigned int size)
{
    unsigned int row = 28;
    unsigned int col = 28;
    unsigned int fnum = row * col;
    FILE* file = fopen(name, "w");

    for (unsigned int i = 0; i < 10000; i++) {
        for (unsigned int j = 0; j < fnum; j++) {
            double val;
            val = img[i][j] / 255.0;

            fprintf(file, "%lf ", val);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main()
{
    mnist_image training_set;
    mnist_label training_label;
    mnist_image test_set;
    mnist_label test_label;
    ffnn nnet;
    trainer learner;

    training_label.read_file("train-labels.idx1-ubyte");
    training_set.read_file("train-images.idx3-ubyte");
    test_label.read_file("t10k-labels.idx1-ubyte");
    test_set.read_file("t10k-images.idx3-ubyte");

    /* train neural network */
    unsigned int size = training_label.get_size();
    unsigned char* label = training_label.get_label();
    unsigned int row = training_set.get_row_size();
    unsigned int col = training_set.get_column_size();
    auto imageset = training_set.get_image();
    unsigned int fnum = row * col;
    double* input = new double[fnum];
    unsigned int iter_max = 10;
    unsigned int layer_size[] = { fnum, 50, 10 };

    nnet.initialize(layer_size, 3);
    initialize_parameter(nnet.get_parameter(), layer_size, 3);

    learner.initialize(&nnet);
    learner.set_activation_function(std::tanh, 1);
    learner.set_derivative_function(tanh_derivative, 1);
    learner.set_activation_function(identity, 2);
    learner.set_derivative_function(identity_derivative, 2);
    learner.set_output_function(softmax);
    learner.set_cost_function(log_cost);
    learner.set_error_init_function(log_error_init);

    double rate = 1.0;

    for (unsigned int iter = 0; iter < 10; iter++) {
        double cost = 0.0;
        double j;
        for (unsigned int i = 0; i < size; i++) {
            unsigned char* image = imageset[i];

            for (unsigned int j = 0; j < fnum; j++)
                input[j] = image[j] / 255.0;

            j = learner.train(input, label[i]);
            learner.update_parameter(rate, 0.0, 0.0);
            cost += j;
        }

        cost = cost / size;
        rate *= 0.9;

        std::cout << "epoch " << iter << " completed! cost:";
        std::cout << cost << std::endl;
    }

    /* test */
    size = test_set.get_size();
    label = test_label.get_label();
    imageset = test_set.get_image();

    unsigned int correct = 0;

    for (unsigned int i = 0; i < size; i++) {
        unsigned char* image = imageset[i];
        unsigned int gold = label[i];
        unsigned int out = 0;
        double* a;
        double max = 0.0;

        for (unsigned int j = 0; j < fnum; j++) {
            input[j] = image[j] / 255.0;
        }

        learner.forward(input);
        a = learner.get_activation(2);

        for (unsigned int j = 0; j < 10; j++)
            if (a[j] > max) {
                max = a[j];
                out = j;
            }

        if (out == gold)
            correct++;
    }

    double precision = (double) correct / size;

    std::cout << "precision: " << precision << std::endl;
}
