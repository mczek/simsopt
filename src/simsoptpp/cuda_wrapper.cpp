#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;

#include <Eigen/Core>

#include "magneticfield.h"
#include "boozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"
#include "tracing.h"

#include <cuda_runtime.h>

namespace py = pybind11;

extern "C" void addKernelWrapper(int *c, const int *a, const int *b, int size);

extern "C" void gpu_tracing(shared_ptr<MagneticField<xt::pytensor>> field, py::array_t<double> xyz_init,
        double m, double q, double vtotal, py::array_t<double> vtang, double tmax, double tol, bool vacuum,
        vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria, int nparticles);
// PYBIND11_MODULE(cuda_module, m) {
//     m.def("add_kernel", [](py::array_t<int> a, py::array_t<int> b){
//         auto a_buf = a.request(), b_buf = b.request();
//         int size = a_buf.size;
//         py::array_t<int> c(size);
//         auto c_buf = c.request();
//         addKernelWrapper((int *)c_buf.ptr, (const int *)a_buf.ptr, (const int *)b_buf.ptr, size);
//         return c;
//     });
// }