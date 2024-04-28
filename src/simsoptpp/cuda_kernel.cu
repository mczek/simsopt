// #include "simdhelpers.h" // import above cuda_runtime to prevent collision for rsqrt
#include <cuda_runtime.h>
#include <iostream>
#include "tracing.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;
namespace py = pybind11;

#include <Eigen/Core>

#include "magneticfield.h"
#include "boozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"


__global__ void addKernel(int *c, const int* a, const int* b, int size){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size){
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void addKernelWrapper(int *c, const int *a, const int *b, int size){
    int *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, size*sizeof(int));
    cudaMalloc((void **)&d_b, size*sizeof(int));
    cudaMalloc((void **)&d_c, size*sizeof(int));

    cudaMemcpy(d_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size*sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<1, 256>>>(d_c, d_a, d_b, size);

    for(int i=0; i<size; ++i){
        std::cout << c[i] <<"\n";
    }

    cudaMemcpy(c, d_c, size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

extern "C" void gpu_tracing(shared_ptr<MagneticField<xt::pytensor>> field, py::array_t<double> xyz_init,
        double m, double q, double vtotal, py::array_t<double> vtang, double tmax, double tol, bool vacuum,
        vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria, int nparticles){
    std::cout << "Hello world!\n";
}


