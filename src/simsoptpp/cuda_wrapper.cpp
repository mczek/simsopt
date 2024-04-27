#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include <cuda_runtime.h>

namespace py = pybind11;

extern "C" void addKernelWrapper(int *c, const int *a, const int *b, int size);

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