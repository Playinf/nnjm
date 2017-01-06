#include <map>
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <io.h>
#include <model.h>
#include <vocab.h>

void help()
{
    std::cout << "usage: ";
    std::cout << "test-vector model type begin end output" << std::endl;
    std::cout << std::endl;
    std::cout << "  model: model of nnjm" << std::endl;
    std::cout << "  type: 0 for source, 1 for target" << std::endl;
    std::cout << "  begin: begin index of output vocabulary" << std::endl;
    std::cout << "  end: end index of output vocabulary" << std::endl;
    std::cout << "  output: output json file" << std::endl;
}

void print_vec(int argc, char** argv)
{
    char* inname = argv[1];
    char* outname = argv[5];
    unsigned int type = std::stod(std::string(argv[2]));
    unsigned int sid = std::stod(std::string(argv[3]));
    unsigned int eid = std::stod(std::string(argv[4]));

    infinity::lm::model model;
    infinity::lm::vocab* vocab;
    load_model(inname, &model);

    switch (type) {
    case 0:
        vocab = model.get_source_vocab();
        break;
    case 1:
        vocab = model.get_target_vocab();
        break;
    default:
        vocab = nullptr;
        break;
    };

    // build id to string map
    std::map<unsigned int, std::string> idmap;
    auto begin = vocab->begin();
    auto end = vocab->end();
    auto iter = begin;
    std::ofstream ofile;

    while (iter != end) {
        std::string word = *iter->first->get_name();
        unsigned int id = iter->second;

        if (word == "\"")
            word = "-QUOTE-";

        idmap[id] = word;
        iter++;
    }

    ofile.open(outname);

    if (ofile.fail()) {
        std::cerr << "cannot open " << outname << std::endl;
        std::exit(-1);
    }

    ofile << "{ \"words\": [" << std::endl;

    auto iditer = idmap.begin();

    // skip
    for (unsigned int i = 0; i < sid; i++)
        iditer++;

    for (unsigned int i = sid; i < eid; i++) {
        std::string word = iditer->second;
        ofile << "\"" << word << "\"";

        if (i + 1 != eid)
            ofile << "," << std::endl;

        iditer++;
    }

    double* embedding = model.get_embedding();
    unsigned int dim = model.get_feature_size();
    iter = begin;

    ofile << "], " << "\"vecs\": [" << std::endl;

    iditer = idmap.begin();

    for (unsigned int i = 0; i < sid; i++)
        iditer++;

    for (unsigned int i = sid; i < eid; i++) {
        unsigned int id = iditer->first;
        double* vec = embedding + id * dim;

        ofile << "[" << std::endl;

        for (unsigned int j = 0; j < dim; j++) {
            ofile << vec[j];

            if (j != dim - 1)
                ofile << ",";
        }

        ofile << "]";

        if (i + 1 != eid)
            ofile << "," << std::endl;
        iditer++;
    }

    ofile << "]}" << std::endl;

    ofile.close();
}

void print_word_vec(int argc, char** argv)
{
    char* inname = argv[1];
    char* outname = argv[5];
    unsigned int sid = std::stod(std::string(argv[3]));
    unsigned int eid = std::stod(std::string(argv[4]));

    infinity::lm::model model;
    infinity::lm::vocab* svocab;
    infinity::lm::vocab* tvocab;
    load_model(inname, &model);

    svocab = model.get_source_vocab();
    tvocab = model.get_target_vocab();

    // build id to string map
    std::map<unsigned int, std::string> sidmap;
    auto sbegin = svocab->begin();
    auto send = svocab->end();
    auto iter = sbegin;

    while (iter != send) {
        std::string word = *iter->first->get_name();
        unsigned int id = iter->second;

        if (word == "\"")
            word = "-QUOTE-";

        sidmap[id] = word;
        iter++;
    }

    std::map<unsigned int, std::string> tidmap;
    auto tbegin = tvocab->begin();
    auto tend = tvocab->end();
    iter = tbegin;

    while (iter != tend) {
        std::string word = *iter->first->get_name();
        unsigned int id = iter->second;

        if (word == "\"")
            word = "-QUOTE-";

        tidmap[id] = word;
        iter++;
    }

    std::ofstream ofile;
    ofile.open(outname);

    if (ofile.fail()) {
        std::cerr << "cannot open " << outname << std::endl;
        std::exit(-1);
    }

    ofile << "{ \"words\": [" << std::endl;

    auto iditer = sidmap.begin();

    // skip source
    for (unsigned int i = 0; i < sid; i++)
        iditer++;

    for (unsigned int i = sid; i < eid; i++) {
        std::string word = iditer->second;
        ofile << "\"" << word << "\"";
        ofile << "," << std::endl;

        iditer++;
    }

    iditer = tidmap.begin();

    for (unsigned int i = 0; i < sid; i++)
        iditer++;

    for (unsigned int i = sid; i < eid; i++) {
        std::string word = iditer->second;
        ofile << "\"" << word << "\"";

        if (i + 1 != eid)
            ofile << "," << std::endl;

        iditer++;
    }

    double* embedding = model.get_embedding();
    unsigned int dim = model.get_feature_size();

    ofile << "], " << "\"vecs\": [" << std::endl;

    iditer = sidmap.begin();

    for (unsigned int i = 0; i < sid; i++)
        iditer++;

    for (unsigned int i = sid; i < eid; i++) {
        unsigned int id = iditer->first;
        double* vec = embedding + id * dim;

        ofile << "[" << std::endl;

        for (unsigned int j = 0; j < dim; j++) {
            ofile << vec[j];

            if (j != dim - 1)
                ofile << ",";
        }

        ofile << "]";
        ofile << "," << std::endl;
        iditer++;
    }

    iditer = tidmap.begin();

    for (unsigned int i = 0; i < sid; i++)
        iditer++;

    for (unsigned int i = sid; i < eid; i++) {
        unsigned int id = iditer->first;
        double* vec = embedding + id * dim;

        ofile << "[" << std::endl;

        for (unsigned int j = 0; j < dim; j++) {
            ofile << vec[j];

            if (j != dim - 1)
                ofile << ",";
        }

        ofile << "]";

        if (i + 1 != eid)
            ofile << "," << std::endl;
        iditer++;
    }

    ofile << "]}" << std::endl;

    ofile.close();
}

int main(int argc, char** argv)
{
    if (argc != 6) {
        help();
        return 1;
    }

    unsigned int type = std::stod(std::string(argv[2]));

    switch (type) {
    case 0:
    case 1:
        print_vec(argc, argv);
        break;
    case 2:
        print_word_vec(argc, argv);
        break;
    default:
        break;
    }
}
