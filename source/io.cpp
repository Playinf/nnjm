/* io.cpp */
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <io.h>
#include <ffnn.h>
#include <misc.h>
#include <model.h>
#include <vocab.h>
#include <utility.h>
#include <parameter.h>

namespace infinity {
namespace lm {

typedef std::map<std::string, std::vector<std::string>> map_type;

void output_parameter(parameter* p, const std::string& n, std::ofstream& f)
{
    int type;
    double dval;
    std::string sval;
    unsigned int ival;
    std::vector<double> dvec;
    std::vector<std::string> svec;
    std::vector<unsigned int> ivec;

    p->get_type(n, type);

    if (type & parameter::type::vector) {
        if (type & parameter::type::real) {
            p->get_parameter(n, dvec);
            f << n << ":";

            for (unsigned int i = 0; i < dvec.size(); i++) {
                f << " " << dvec[i];
            }

            f << '\n';
        }

        if (type & parameter::type::string) {
            p->get_parameter(n, svec);
            f << n << ":";

            for (unsigned int i = 0; i < svec.size(); i++) {
                f << " " << svec[i];
            }

            f << '\n';
        }

        if (type & parameter::type::integer) {
            p->get_parameter(n, ivec);
            f << n << ":";

            for (unsigned int i = 0; i < ivec.size(); i++) {
                f << " " << ivec[i];
            }

            f << '\n';
        }
    } else {
        if (type & parameter::type::real) {
            p->get_parameter(n, dval);
            f << n << ": " << dval << '\n';
        }

        if (type & parameter::type::string) {
            p->get_parameter(n, sval);
            f << n << ": " << sval << '\n';
        }

        if (type & parameter::type::integer) {
            p->get_parameter(n, ival);
            f << n << ": " << ival << '\n';
        }
    }
}

// save model parameter to model file
static void save(parameter* param, std::ofstream& file)
{
    unsigned int pnum = 25;

    file << pnum << '\n';
    output_parameter(param, "model", file);
    output_parameter(param, "order", file);
    output_parameter(param, "window", file);
    output_parameter(param, "source-context", file);
    output_parameter(param, "target-context", file);

    // neural network architecture
    output_parameter(param, "network", file);
    output_parameter(param, "layer-number", file);
    output_parameter(param, "feature-number", file);
    output_parameter(param, "input-size", file);
    output_parameter(param, "hidden-size", file);
    output_parameter(param, "output-size", file);
    output_parameter(param, "activation-function", file);
    output_parameter(param, "output-function", file);

    // training options
    output_parameter(param, "learning-rate", file);
    output_parameter(param, "weight-decay", file);
    output_parameter(param, "momentum", file);
    output_parameter(param, "norm-control", file);
    output_parameter(param, "epoch", file);
    output_parameter(param, "max-epoch", file);
    output_parameter(param, "batch-size", file);
    output_parameter(param, "init-method", file);
    output_parameter(param, "init-range", file);
    output_parameter(param, "update-range", file);
    output_parameter(param, "weight-range", file);
    output_parameter(param, "cost-function", file);
}

// save matrix to model file
static void save(double* a, unsigned int m, unsigned int n, std::ofstream& f)
{
    unsigned int pnum;

    f.write(reinterpret_cast<const char*>(&m), sizeof(unsigned int));
    f.write(reinterpret_cast<const char*>(&n), sizeof(unsigned int));

    pnum = m * n;

    for (unsigned int i = 0; i < pnum; i++) {
        const char* ptr = reinterpret_cast<const char*>(a + i);
        f.write(ptr, sizeof(double));
    }
}

// save vocabulary to model file
static void save(vocab* v, std::ofstream& f)
{
    unsigned int num;
    char* ptr = reinterpret_cast<char*>(&num);

    num = v->get_size();
    f.write(ptr, sizeof(unsigned int));

    auto begin = v->begin();
    auto end = v->end();

    for (auto iter = begin; iter != end; iter++) {
        const symbol* sym = iter->first;
        unsigned int id = iter->second;
        f << *sym->get_name() << " ||| " << id << "\n";
    }
}

// save model to model file
static void save(model* m, std::ofstream& file)
{
    ffnn* network = m->get_network();
    parameter* param = m->get_parameter();
    unsigned int input_size;
    unsigned int feature_size;
    unsigned int layer_number;
    double* weight = network->get_parameter();
    double* p = weight;

    // save parameter
    save(param, file);

    // save vocabulary
    save(m->get_source_vocab(), file);
    save(m->get_target_vocab(), file);
    save(m->get_output_vocab(), file);

    // get necessary information
    input_size = m->get_input_size();
    feature_size = m->get_feature_size();

    // save embedding
    save(m->get_embedding(), input_size, feature_size, file);

    layer_number = network->get_layer_number();

    // save network synapse
    for (unsigned int i = 1; i < layer_number; i++) {
        unsigned int inum = network->get_layer_size(i - 1) + 1;
        unsigned int onum = network->get_layer_size(i);
        save(p, inum, onum, file);
        p += inum * onum;
    }
}

// load parameter map from model file
static void load(map_type& map, std::ifstream& file)
{
    std::string line;
    unsigned int param_number;

    std::getline(file, line);
    param_number = std::atoi(line.c_str());

    for (unsigned int i = 0; i < param_number; i++) {
        std::vector<std::string> vec;
        std::vector<std::string> val;
        std::string name;

        std::getline(file, line);
        string_split(line, ":", vec);
        string_trim(vec[0], name);
        string_split(vec[1], " ", val);
        map[name] = val;
    }
}

// load vocabulary from model file
static void load(vocab* v, std::ifstream& f)
{
    std::string line;
    unsigned int num;
    char* ptr = reinterpret_cast<char*>(&num);

    f.read(ptr, sizeof(unsigned int));

    for (unsigned int i = 0; i < num; i++) {
        unsigned int id;
        std::string name;
        std::string line;
        std::vector<std::string> vec;

        std::getline(f, line);
        string_split(line, "|||", vec);
        string_trim(vec[0], name);
        id = std::stoi(vec[1]);
        v->insert(name, id);
    }
}

// load network synapse from model file
static void load(double* a, std::ifstream& file)
{
    unsigned int row;
    unsigned int col;
    unsigned int pnum;
    char* prow = reinterpret_cast<char*>(&row);
    char* pcol = reinterpret_cast<char*>(&col);

    file.read(prow, sizeof(unsigned int));
    file.read(pcol, sizeof(unsigned int));

    pnum = row * col;

    for (unsigned int i = 0; i < pnum; i++) {
        char* ptr = reinterpret_cast<char*>(a + i);
        file.read(ptr, sizeof(double));
    }
}

// load model from model file
void load(model* m, std::ifstream& file)
{
    map_type param_map;
    parameter* param = m->get_parameter();

    // load parameters
    load(param_map, file);
    // convert parameter
    load_parameter(param_map, param);
    // initialize
    m->initialize();
    // load vocabulary
    load(m->get_source_vocab(), file);
    load(m->get_target_vocab(), file);
    load(m->get_output_vocab(), file);
    // load embeddings
    load(m->get_embedding(), file);

    // load synapse
    ffnn* net = m->get_network();
    double* weight = net->get_parameter();
    unsigned int layer_number = net->get_layer_number();

    for (unsigned int i = 1; i < layer_number; i++) {
        unsigned int inum = net->get_layer_size(i - 1) + 1;
        unsigned int onum = net->get_layer_size(i);
        load(weight, file);
        weight += inum * onum;
    }
}


void save_model(const char* n, model* m)
{
    std::ofstream file;

    file.open(n, std::ios::binary);

    if (file.fail()) {
        std::string msg;
        msg = "cannot save model to " + std::string(n);
        throw std::runtime_error(msg);
    }

    save(m, file);
}

void load_model(const char* n, model* m)
{
    std::ifstream file;

    file.open(n, std::ios::binary);

    if (file.fail()) {
        std::string msg;
        msg = "cannot open model from " + std::string(n);
        throw std::runtime_error(msg);
    }

    load(m, file);
}

void load_vocab(const char* n, vocab* v)
{
    std::ifstream file;

    file.open(n);

    if (file.fail()) {
        std::string msg;
        msg = "cannot open vocabulary from " + std::string(n);
        throw std::runtime_error(msg);
    }

    std::string line;

    while (std::getline(file, line)) {
        std::vector<std::string> vec;
        std::string name;
        unsigned int id;

        string_split(line, "|||", vec);

        string_trim(vec[0], name);
        id = std::stoi(vec[1]);
        v->insert(name, id);
    }
}

} /* lm */
} /* infinity */
