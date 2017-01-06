#include <Eigen/Dense>
#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/prod.hpp>
#include <ctime>
#include <CL/cl.h>
#include <mathlib.h>
#include <thread>

template<class T>
class device {
public:
    typedef unsigned int size_type;
    typedef cl_mem device_memory;
    typedef typename viennacl::backend::mem_handle handle_type;
    typedef typename viennacl::context context_type;

    static size_type get_aligned_size(size_type size)
    {
        size_type padding_size = viennacl::dense_padding_size;
        size = viennacl::tools::align_to_multiple(size, padding_size);

        return size;
    }
/*
    static device_memory create_memory(context_type& ctx, size_type size)
    {
        cl_int err;
        device_memory mem;
        size_type flags = CL_MEM_READ_WRITE;

        mem = clCreateBuffer(ctx.get(), flags, size, nullptr, &err);

        if (err != CL_SUCCESS)
            throw std::runtime_error("allocate device memory failed");

        return mem;
    }

    static device_memory create_vector_memory(size_type n)
    {
        size_type size = get_aligned_size(n) * sizeof(T)
        return create_memory(viennacl::context().get(), size);
    }

    static device_memory create_matrix_memory(size_type m, size_type n)
    {
        size_type size = get_aligned_size(m) * get_aligned_size(n) * sizeof(T);

        return create_memory(viennacl::context().get(), size);
    }
*/
    static device_memory sub_memory(device_memory m, size_type o, size_type s)
    {
        cl_int err;
        size_type flags = CL_MEM_READ_WRITE;
        size_type type = CL_BUFFER_CREATE_TYPE_REGION;
        cl_buffer_region region;

        region.origin = o;
        region.size = s;

        clCreateSubBuffer(m, flags, type, &region, &err);

        if (err != CL_SUCCESS) {
            if (err == CL_MISALIGNED_SUB_BUFFER_OFFSET)
                throw std::runtime_error("misaligned sub buffer");
            throw std::runtime_error("allocate sub memory failed");
        }
    }
};

void test_sigmoid()
{
    std::string my_program =
    "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "kernel void elementwise_sigmoid(global double* v, unsigned int s)\n"
    "{\n"
    "    for (unsigned int i = get_global_id(0); i < s; i++)\n"
    "        v[i] = 1.0 / (1.0 + exp(v[i]));\n"
    "}\n";
    viennacl::ocl::program& pgm = viennacl::ocl::current_context().add_program(my_program, "my_program");
    viennacl::ocl::kernel& my_kernel_sigmoid = pgm.get_kernel("elementwise_sigmoid");
    my_kernel_sigmoid.local_work_size(0, 16);
    my_kernel_sigmoid.global_work_size(0, 128);

    viennacl::vector<double> vec(32000);

    for (unsigned int i = 0; i < 1000; i++) {

        viennacl::ocl::enqueue(my_kernel_sigmoid(vec, static_cast<cl_uint>(vec.size())));
    }
}

void sigmoid(std::vector<double>& vec)
{
    unsigned int size = vec.size();

    for (unsigned int i = 0; i < size; i++) {
        vec[i] = 1.0 / (1.0 + vec[i]);
    }
}

void test_copy()
{
    auto t1 = std::clock();

    viennacl::vector<double> xmat(500);
    std::vector<double> vec(500);

    for (unsigned int i = 0; i < 1000; i++) {
        viennacl::linalg::element_exp(xmat);
        viennacl::copy(xmat, vec);
        sigmoid(vec);
        viennacl::copy(vec, xmat);
    }

    auto t2 = std::clock();
    std::cout << 1/((double)(t2 - t1) / CLOCKS_PER_SEC / 1000) << std::endl;
}

void test_copy2()
{
    auto t1 = std::clock();

    viennacl::vector<double> xmat(32000);
    std::vector<double> xvec(32000);

    for (unsigned int i = 0; i < 1000; i++) {
        viennacl::copy(xvec, xmat);
        viennacl::copy(xmat, xvec);
    }

    auto t2 = std::clock();
    std::cout << 1/((double)(t2 - t1) / CLOCKS_PER_SEC / 1000) << std::endl;
}


void test1()
{
    auto t1 = std::clock();

    viennacl::matrix<double> vmat(32000, 200);
    viennacl::matrix<double> wmat(32000, 500);
    viennacl::matrix<double> smat(3000, 500);
    viennacl::vector<double> xmat(500);
    viennacl::vector<double> ymat(32000);
    viennacl::vector<double> imat(3000);

    std::cout << vmat.internal_size() << std::endl;
    std::cout << wmat.internal_size() << std::endl;
    std::cout << imat.internal_size() << std::endl;
    std::cout << xmat.internal_size() << std::endl;
    std::cout << ymat.internal_size() << std::endl;

    for (unsigned int i = 0; i < 1000; i++)
    {
        imat = viennacl::linalg::prod(viennacl::trans(smat), xmat);
        //xmat = viennacl::linalg::prod(viennacl::trans(wmat), ymat);
    }

    auto t2 = std::clock();
    Eigen::setNbThreads(1);

    Eigen::Matrix<float, -1, -1> ewmat(32000, 500);
    Eigen::Matrix<float, -1, -1> exmat(1, 500);
    Eigen::Matrix<float, -1, -1> eymat(1, 32000);

    for (unsigned int i = 0; i < 1000; i++)
    {
        for (unsigned int j = 0; j < 500; j++) {
            eymat.noalias() = ewmat * exmat;
        }
    }

    auto t3 = std::clock();

    std::cout << 1/((double)(t2 - t1) / CLOCKS_PER_SEC / 1000) << std::endl;
    std::cout << 1/((double)(t3 - t2) / CLOCKS_PER_SEC / 1000) << std::endl;
}

void parallel_matxvec(double* d, double* a, double* v, unsigned int m, unsigned int n)
{
    unsigned int size[4];
    size[0] = m / 4;
    size[1] = m / 4;
    size[2] = m / 4;
    size[3] = m / 4;
    std::thread t1 {matxvec, d, a, v, size[0], n};
    std::thread t2 {matxvec, d + size[0], a + size[0] * n, v, size[1], n};
    std::thread t3 {matxvec, d + 2 * size[0], a + 2 * size[0] * n, v, size[2], n};
    std::thread t4 {matxvec, d + 3 * size[0], a + 3 * size[0] * n, v, size[3], n};

    t1.join();
    t2.join();
    t3.join();
    t4.join();

}

void test_cpu()
{
    auto t1 = std::clock();

    double* yvec = new double[32000];
    double* xvec = new double[501];
    double* wmat = new double[32000 * 501];

    for (unsigned int i = 0; i < 1000; i++) {
        matxvec(yvec, wmat, xvec, 32000, 501);
    }

    auto t2 = std::clock();

    std::cout << 1/((double)(t2 - t1) / CLOCKS_PER_SEC / 1000) << std::endl;
}

void test_neural()
{
    std::vector<double> input(3001);
    std::vector<double> hidden(501);
    std::vector<double> output(32000);
    viennacl::vector<double> gpu_input(3001);
    viennacl::vector<double> gpu_hidden(501);
    viennacl::vector<double> gpu_output(32000);
    viennacl::matrix<double> w1(3001, 500);
    viennacl::matrix<double> w2(501, 32000);

    auto t1 = std::clock();

    for (unsigned int i = 0; i < 1000; i++) {
        viennacl::copy(input, gpu_input);
        //gpu_hidden = viennacl::linalg::prod(viennacl::trans(w1), gpu_input);
        viennacl::copy(gpu_hidden, hidden);
        viennacl::copy(hidden, gpu_hidden);
        //gpu_output = viennacl::linalg::prod(viennacl::trans(w2), gpu_hidden);
        viennacl::copy(gpu_output, output);
        viennacl::copy(output, gpu_output);
        viennacl::copy(gpu_hidden, hidden);
        viennacl::copy(hidden, gpu_hidden);
    }

    auto t2 = std::clock();

    std::cout << 1/((double)(t2 - t1) / CLOCKS_PER_SEC / 1000) << std::endl;
}

void test_grad()
{
    auto t1 = std::clock();

    double* v1 = new double[32000];
    double* v2 = new double[500];
    double* v3 = new double[32000 * 500];

    #pragma omp parallel for
    for (unsigned int i = 0; i < 1000; i++) {
        #pragma omp parallel for
        for (unsigned int j = 0; j < 32000; j++)
            for (unsigned int k = 0; k < 500; k++)
                v3[i * 500 + j] = v1[j] * v2[k];
    }

    auto t2 = std::clock();
    std::cout << 1/((double)(t2 - t1) / CLOCKS_PER_SEC / 1000) << std::endl;
}

int main()
{
    test_cpu();
}
/*
  explicit matrix(cl_mem mem, size_type rows, size_type internal_row_count,
                  size_type cols, size_type internal_col_count)
    : base_type(mem, viennacl::context(),
                rows, 0, 1, internal_row_count,
                cols, 0, 1, internal_col_count,
                viennacl::is_row_major<F>::value) {}
*/
