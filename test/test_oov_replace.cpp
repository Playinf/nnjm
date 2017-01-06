#include <iostream>
#include <nnjm.h>

int main()
{
    infinity::lm::nnjm model;

    model.load("model-iter-3.nnjm");
    model.load_oov_map("replace.txt");
}
