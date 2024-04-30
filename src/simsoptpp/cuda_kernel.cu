// #include "simdhelpers.h" // import above cuda_runtime to prevent collision for rsqrt
#include <cuda_runtime.h>
#include <iostream>
#include "tracing.h"
#include <math.h>
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


// Particle Data Structure
typedef struct particle_t {
    double x;  // Position X
    double y;  // Position Y
    double z;  // Position Z
    double v_par; // Velocity parallel
    double v_perp; // Velocity perpendicular
    double dotx;
    double doty;
    double dotz;
    double dotv_par;
    bool has_left;
} particle_t;

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



__global__ void particle_trace_kernel(MagneticField<xt::pytensor> field, const double* xyz_init_arr,
        double m, double q, double vtotal, const double* vtang_arr, double tmax, double tol, bool vacuum, int nparticles, 
        tuple<vector<array<double, 5>>, vector<array<double, 6>>>* out){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){

        int start = 3*idx;
        // array<double, 3> xyz_init_i = {xyz_init_arr[start], xyz_init_arr[start+1], xyz_init_arr[start+2]};
        // typename MagneticField<xt::pytensor>::Tensor2 xyz({{xyz_init_arr[start], xyz_init_arr[start+1], xyz_init_arr[start+2]}});
        // field.set_points(xyz);
        
    //     out[idx] = particle_guiding_center_tracing(field, xyz_init_i, m, q, vtotal, vtang_arr[idx], tmax, tol, vacuum, phis, stopping_criteria);
    //     // res_all[idx] = std::get<0>(out_i);
    //     // res_phi_hits_all[idx] = std::get<1>(out_i);
    }
}

void shape(double x, double* shape){
    shape[0] = (1.0-x)*(2.0-x)*(3.0-x)/6.0;
    shape[1] = x*(2.0-x)*(3.0-x)/2.0;
    shape[2] = x*(x-1.0)*(3.0-x)/2.0;
    shape[3] = x*(x-1.0)*(x-2.0)/6.0;
    return;         
}

void dshape(double x, double h, double* dshape){
    dshape[0] = (-(2.0-x)*(3.0-x)-(1.0-x)*(3.0-x)-(1.0-x)*(2.0-x))/(h*6.0);
    dshape[1] = ( (2.0-x)*(3.0-x)-x*(3.0-x)-x*(2.0-x))/(h*2.0);
    dshape[2] = ( (x-1.0)*(3.0-x)+x*(3.0-x)-x*(x-1.0))/(h*2.0);
    dshape[3] = ( (x-1.0)*(x-2.0)+x*(x-2.0)+x*(x-1.0))/(h*6.0);
    return;         
}



extern "C" tuple<vector<vector<array<double, 5>>>, vector<vector<array<double, 6>>>> gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<double> xyz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, bool vacuum, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria, int nparticles){

    vector<vector<array<double, 5>>> res_all(nparticles);
    vector<vector<array<double, 6>>> res_phi_hits_all(nparticles);


    //  read data in from python
    py::buffer_info xyz_buf = xyz_init.request();
    double* xyz_init_arr = static_cast<double*>(xyz_buf.ptr);

    // contsins b field and then curve distance
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info r_buf = rrange.request();
    double* rrange_arr = static_cast<double*>(r_buf.ptr);
    double r_grid_size = (rrange_arr[1] - rrange_arr[0]) / (rrange_arr[2]-1);

    py::buffer_info phi_buf = phirange.request();
    double* phirange_arr = static_cast<double*>(phi_buf.ptr);
    double phi_grid_size = 2*M_PI / phirange_arr[2];

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);
    double z_grid_size = (zrange_arr[1] - zrange_arr[0]) / (zrange_arr[2]-1);


    particle_t particles[nparticles];
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;
        particles[i].x = xyz_init_arr[start];
        particles[i].y = xyz_init_arr[start+1];
        particles[i].z = xyz_init_arr[start+2];
        particles[i].v_par = xyz_init_arr[start+3];
        particles[i].v_perp = vtotal*vtotal -  particles[i].v_par* particles[i].v_par;
        particles[i].has_left = false;
    }

    std::cout << "particles initialized \n";
    double dt = 0.00001;
    int nsteps = (int) (tmax / dt);
    double r_shape[4];
    double phi_shape[4];
    double z_shape[4];

    double r_dshape[4];
    double phi_dshape[4];
    double z_dshape[4];

    double B[3];
    double grad_B[9];
    for(int p=0; p<nparticles; ++p){
        for(int time_step=0; time_step<nsteps; ++time_step){

            /*
             * Time step ODE
             */
            double x = particles[p].x;
            double y = particles[p].y;
            double z = particles[p].z;

            // magnetic field quad points are in cylindrical coordinates
            double r = sqrt(x*x + y*y);
            double phi = atan2(y, x);

            // index into mesh to obtain nearby points
            int i = (int) ((r - rrange_arr[0]) / r_grid_size);
            int j = (int) ((z - zrange_arr[0]) / z_grid_size);
            int k = (int) ((phi - phirange_arr[0]) / phi_grid_size);


            // normalized positions in local grid wrt e.g. r at index i
            int nr = rrange_arr[2];
            int nphi = phirange_arr[2];
            int nz = zrange_arr[2];
            double r_rel = (r -  ((rrange_arr[1]*(i-1) + rrange_arr[0]*(nr-i)) / (nr-1))) / r_grid_size;
            double z_rel = (z -  ((zrange_arr[1]*(k-1) + zrange_arr[0]*(nz-k)) / (nz-1))) / z_grid_size;
            double phi_rel = (phi - M_PI*(((k-1) % nphi) / nphi  - (nphi - 1 - ((k-1) % nphi))/(nphi-1))) / phi_grid_size;

            std::cout << "grid point found \n";

            shape(r_rel, r_shape);
            shape(phi_rel, phi_shape);
            shape(z_rel, z_shape);

            std::cout <<"shape set \n";
            // accumulate interpolation of B
            B[0] = 0.0;
            B[1] = 0.0;            
            B[2] = 0.0;

            // interpolate the distance to the surface
            double surface_dist = 0.0;

            std::cout << "starting B accumulation\n";
            // quad pts are indexed r z phi
            for(int ii=-1; ii<=2; ++ii){             
                for(int jj=-1; jj<=2; ++jj){                 
                    for(int kk=-1; kk<=2; ++kk){
                        int wrap_k = (k+kk-1) % nphi + 1;
                        if ((i+ii >= 0 & i+ii < nr) & (j+jj >= 0 & j+jj < nz)){
                            int start = 4*((i+ii)*nz*nphi + (j+jj)*nphi + (wrap_k));
                            std::cout << "start=" << start << "\t" << 4*nr*nz*nphi << "\n";
                            B[0] += quadpts_arr[start] * r_shape[ii+1]*z_shape[jj+1]*phi_shape[kk+1];
                            B[1] += quadpts_arr[start+1] * r_shape[ii+1]*z_shape[jj+1]*phi_shape[kk+1];
                            B[2] += quadpts_arr[start+2] * r_shape[ii+1]*z_shape[jj+1]*phi_shape[kk+1];
                        
                            surface_dist += quadpts_arr[start+3] * r_shape[ii+1]*z_shape[jj+1]*phi_shape[kk+1];
                        }

                    }
                }
            }
            std::cout << "B interpolated \n";


            particles[p].has_left = (particles[p].has_left || surface_dist < 0);
            if(particles[p].has_left){
                break;
            }

            //  accumulate grad B: rows are partial deriv wrt r, z, phi
            //  row major order
            for(int m=0; m<9; ++m){
                grad_B[m] = 0.0;
            }
            dshape(r_rel, r_grid_size, r_dshape);
            dshape(phi_rel, phi_grid_size, phi_dshape);
            dshape(z_rel, z_grid_size, z_dshape);

            for(int ii=-1; ii<=2; ++ii){             
                for(int jj=-1; jj<=2; ++jj){                 
                    for(int kk=-1; kk<=2; ++kk){
                        int wrap_k = (k+kk-1) % nphi + 1;
                        int start = 4*((i+ii)*nz*nphi + (j+jj)*nphi + (wrap_k));
                        if(start > 0 && start < 4*nr*nz*nphi){ // if valid index
                            // interpolate gradient for each entry of B, filling in each column of the gradient
                            for(int l=0; l<3; ++l){
                                double Bval = quadpts_arr[start+l];
                                grad_B[l] += Bval * r_dshape[ii+1]*z_shape[jj+1]*phi_shape[kk+1];
                                grad_B[3+l] += Bval * r_shape[ii+1]*z_dshape[jj+1]*phi_shape[kk+1];
                                grad_B[6+l] += Bval * r_shape[ii+1]*z_shape[jj+1]*phi_dshape[kk+1];
                            }
                        }

                    }
                }
            }

            std::cout << "grad B interpolated \n";


            // convert gradient from cylindrical to cartesian coordinates x,y,z
            double c = cos(phi);
            double s = sin(phi);


            for(int l=0; l<3; ++l){ // iter over column
                double dfdr = grad_B[l];
                double dfdphi_divr = grad_B[l+3] / r;
                
                grad_B[l] = c*dfdr - s*dfdphi_divr;
                grad_B[l+6] = grad_B[l+3]; // z index changes
                grad_B[l+3] = s*dfdr + c*dfdphi_divr;
            }


            // take ODE step

            std::cout << "starting updates \n";

            double normB = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
            double nabla_normB[3];

            // use transpose of grad_B to compute \nabla B \dot B
            nabla_normB[0] = (grad_B[0]*B[0] + grad_B[3]*B[1] + grad_B[6]*B[2]) / (2*normB);
            nabla_normB[1] = (grad_B[1]*B[0] + grad_B[4]*B[1] + grad_B[7]*B[2]) / (2*normB);
            nabla_normB[2] = (grad_B[2]*B[0] + grad_B[5]*B[1] + grad_B[8]*B[2]) / (2*normB);

            double cross_prod[3];
            cross_prod[0] = B[1]*nabla_normB[2] - B[2]*nabla_normB[1];
            cross_prod[1] = B[2]*nabla_normB[0] - B[0]*nabla_normB[2];
            cross_prod[2] = B[0]*nabla_normB[1] - B[1]*nabla_normB[0];


            particles[p].dotx = particles[p].v_par * B[0]/normB + m/(q*pow(normB, 3)) * (0.5*pow(particles[p].v_perp, 2) + pow(particles[p].v_par, 2))*cross_prod[0];
            particles[p].doty = particles[p].v_par * B[1]/normB + m/(q*pow(normB, 3)) * (0.5*pow(particles[p].v_perp, 2) + pow(particles[p].v_par, 2))*cross_prod[1];
            particles[p].dotz = particles[p].v_par * B[2]/normB + m/(q*pow(normB, 3)) * (0.5*pow(particles[p].v_perp, 2) + pow(particles[p].v_par, 2))*cross_prod[2];
            
            double mu = particles[p].v_perp / (2*normB);
            double BdotNablaNormB = B[0]*nabla_normB[0] + B[1]*nabla_normB[1] + B[2]*nabla_normB[2];
            particles[p].dotv_par = -mu*BdotNablaNormB/normB;
            particles[p].v_perp = 2*mu*normB;


            // update
            particles[p].x += particles[p].dotx * dt;
            particles[p].y += particles[p].doty * dt;
            particles[p].z += particles[p].dotz * dt;
            particles[p].v_par += particles[p].dotv_par * dt;

            std::cout << "updates complete \n";

        }
    }



    std::cout << "Hello world!\n";
    return std::make_tuple(res_all, res_phi_hits_all);
}



