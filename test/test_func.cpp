/* test_func.cpp */
#include <iostream>
#include <functional.h>

template <class T>
void func_wrapper(T func, double* arg1, unsigned int arg2)
{
    func(arg1, arg2);
}

template <class T>
void activate(T* func, double* arg1, unsigned int arg2)
{
    func->operator()(arg1, arg2);
}

struct activate_wrapper {
    void operator()(activation_function* func, double* arg1, unsigned int arg2)
    {
        func->operator()(arg1, arg2);
    }
};

void copy(double* d, double* s, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
        d[i] = s[i];
}

void print(double* a, unsigned int n)
{
    for (unsigned int i = 0; i < n; i++)
        std::cout << a[i] << ' ';
    std::cout << std::endl;
}

int main()
{
    double a[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0
    };
    double b[10] = { 0.0 };
    tanh tanh_act;
    sigmoid sigmoid_act;
    identity identity_act;
    activation_function* func[] = {
        &tanh_act,
        &sigmoid_act,
        &identity_act,
    };
    auto lambda = [](activation_function* func, double* d, unsigned int n) { func->operator()(d, n); };

    for (int i = 0; i < 3; i++) {
        copy(b, a, 10);
        //(*func[i])(b, 10);
        //activate(func[i], b, 10);
        //activate_wrapper()(func[i], b, 10);
        lambda(func[i], b, 10);
        print(b, 10);
    }
}
