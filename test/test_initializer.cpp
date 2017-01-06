#include <vector>
#include <string>
#include <iostream>

void test(std::vector<double> vec)
{
    for (unsigned int i = 0; i < vec.size(); i++)
        std::cout << vec[i] << ", ";
    std::cout << std::endl;
}

int main()
{
    auto il = { 0.0, 0.0 };
    std::vector<double> vec(il);

    test(il);
}
