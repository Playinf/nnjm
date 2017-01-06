/*
 * manager.h
 *
 * author: Playinf
 * email: playinf@stu.xmu.edu.cn
 *
 */
#ifndef __MANAGER_H__
#define __MANAGER_H__

#include <string>
#include <vector>
#include <handler.h>

namespace infinity {
namespace lm {

class model;
class vocab;

class manager {
public:
    manager();
    ~manager();

    void train();
    void test();
    void manage(char** args, unsigned int n);
private:
    double test_file(const char* name);
    void make_ngram(std::vector<std::string>& v, unsigned int* n);
private:
    vocab* source_vocab;
    vocab* target_vocab;
    vocab* output_vocab;
    model* neural_model;
    unsigned int order;
    unsigned int input_size;
    unsigned int output_size;
    unsigned int feature_size;
    unsigned int source_context;
    unsigned int target_context;
    unsigned int activation_size;
    output_handler output_function;
    std::string model_filename;
    std::string training_filename;
    std::string testing_filename;
    std::string validation_filename;
};

} /* lm */
} /* infinity */

#endif /* __MANAGER_H__ */
