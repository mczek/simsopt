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

// #include <Eigen/Core>

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

extern "C" tuple<vector<vector<array<double, 5>>>, vector<vector<array<double, 6>>>> gpu_tracing(shared_ptr<MagneticField<xt::pytensor>> field, py::array_t<double> xyz_init,
        double m, double q, double vtotal, py::array_t<double> vtang, double tmax, double tol, bool vacuum,
        vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria, int nparticles){

    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr); 

    py::buffer_info xyz_buf = xyz_init.request();
    double* xyz_init_arr = static_cast<double*>(xyz_buf.ptr); 

    vector<vector<array<double, 5>>> res_all;
    vector<vector<array<double, 6>>> res_phi_hits_all;
    for(int i=0; i<nparticles; ++i){
        std::cout << vtang_arr[i] << "\n";

        // numpy arrays are in row major order
        int start = 3*i;
        array<double, 3> xyz_init_i = {xyz_init_arr[start], xyz_init_arr[start+1], xyz_init_arr[start+2]};
        for(int j=0; j<3; ++j){
            std::cout << xyz_init_i[j] << "\t";
        }
        std::cout << "\n";

        tuple<vector<array<double, 5>>, vector<array<double, 6>>> out_i = particle_guiding_center_tracing(field, xyz_init_i, m, q, vtotal, vtang_arr[i], tmax, tol, vacuum, phis, stopping_criteria);
        res_all.push_back(std::get<0>(out_i));
        res_phi_hits_all.push_back(std::get<1>(out_i));
    }
    


    std::cout << "Hello world!\n";
    return std::make_tuple(res_all, res_phi_hits_all);
}



