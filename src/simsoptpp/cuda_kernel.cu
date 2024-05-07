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

// #define dt 1e-7

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
        // // std::cout << c[i] <<"\n";
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

// state has 5 elements: current values of x,y,z,v_perp
// // compute derivs for: x, y, z, v_perp
// void calc_derivs(double* state, double* derivs, double* rrange_arr, double* zrange_arr, double* phirange_arr, double* quadpts_arr,
//                         double dt, double tmax, double m, double q) {
//     double x = state[0];
//     double y = state[1];
//     double z = state[2];
//     double v_perp = state[3];

//     double r_shape[4];
//     double phi_shape[4];
//     double z_shape[4];

//     double r_dshape[4];
//     double phi_dshape[4];
//     double z_dshape[4];

//     double B[3];
//     double grad_B[9];
//     double nabla_normB[3];
//     double cross_prod[3];

//     double r_grid_size = (rrange_arr[1] - rrange_arr[0]) / (rrange_arr[2]-1);
//     double phi_grid_size = 2*M_PI / phirange_arr[2];
//     double z_grid_size = (zrange_arr[1] - zrange_arr[0]) / (zrange_arr[2]-1);

//     // interpolate B field for current state
//     double r = sqrt(x*x + y*y);
//     double phi = atan2(y, x);

//     // index into mesh to obtain nearby points
//     int i = (int) ((r - rrange_arr[0]) / r_grid_size) + 1;
//     int j = (int) ((z - zrange_arr[0]) / z_grid_size) + 1;
//     int k = (int) (phi / phi_grid_size) + 1;



//     // normalized positions in local grid wrt e.g. r at index i
//     int nr = rrange_arr[2];
//     int nphi = phirange_arr[2];
//     int nz = zrange_arr[2];
//     double r_rel = (r -  (rrange_arr[0] + i*r_grid_size)) / r_grid_size;
//     double z_rel = (z -  (zrange_arr[0] + j*z_grid_size)) / z_grid_size;
//     double phi_rel = (phi - (k*phi_grid_size)) / phi_grid_size;


//     // std::cout << r << "\t" << -1*(r_rel*r_grid_size - r) << "\t" << r_grid_size << "\n";
//     // std::cout << z << "\t" << -1*(z_rel*z_grid_size - z) << "\t" << z_grid_size << "\n";
//     // std::cout << phi << "\t" << -1*(phi_rel*phi_grid_size - phi) << "\t" << phi_grid_size << "\n";
//     // std::cout << "using index " <<  (i*nz*nphi + j*nphi + k) << "\n";
//     // std::cout << quadpts_arr[4*(i*nz*nphi + j*nphi + k) + 3] << "\n";
//     // // std::cout << "grid point found \n";

//     // // std::cout << "r_rel " << r_rel << "\t" << z_rel << "\t" << phi_rel << "\n";

//     shape(r_rel, r_shape);
//     shape(z_rel, z_shape);
//     shape(phi_rel, phi_shape);


//     // // std::cout <<"shape set \n";
//     // accumulate interpolation of B
//     B[0] = 0.0;
//     B[1] = 0.0;            
//     B[2] = 0.0;

//     // interpolate the distance to the surface
//     double surface_dist = 0.0;

//     // // std::cout << "starting B accumulation\n";
//     // quad pts are indexed r z phi
//     bool is_lost = false;
//     for(int ii=0; ii<=3; ++ii){             
//         for(int jj=0; jj<=3; ++jj){                 
//             for(int kk=0; kk<=3; ++kk){
//                 int wrap_k = ((k+kk-1) % nphi) + 1;

//                 if ((i+ii >= 0 & i+ii < nr) & (j+jj >= 0 & j+jj < nz)){
//                     int start = 4*((i+ii)*nz*nphi + (j+jj)*nphi + (wrap_k));
//                     // // std::cout << "start=" << start << "\t" << 4*nr*nz*nphi << "\n";
//                     B[0] += quadpts_arr[start]   * r_shape[ii]*z_shape[jj]*phi_shape[kk];
//                     B[1] += quadpts_arr[start+1] * r_shape[ii]*z_shape[jj]*phi_shape[kk];
//                     B[2] += quadpts_arr[start+2] * r_shape[ii]*z_shape[jj]*phi_shape[kk];

//                     is_lost = is_lost || (quadpts_arr[start+3] < 0); 
//                     // // std::cout << ii << "\t" << jj << "\t" << kk << "\n";
//                     // // std::cout << "interp surface dist val: " << quadpts_arr[start+3] << "\n";
//                     surface_dist += quadpts_arr[start+3] * r_shape[ii]*z_shape[jj]*phi_shape[kk];
//                 } else{
//                     // // std::cout << "bad grid index for" << r << "\t" << phi << "\t" << z <<"\n"; 
//                 }

//             }
//         }
//     }

// }

// out contains derivatives for x , y, z, v_par, and then norm of B and surface distance interpolation
void calc_derivs(double* state, double* out, double* rrange_arr, double* zrange_arr, double* phirange_arr, double* quadpts_arr, double m, double q, double mu){
    double r_shape[4];
    double phi_shape[4];
    double z_shape[4];

    double r_dshape[4];
    double phi_dshape[4];
    double z_dshape[4];

    double B[3];
    double grad_B[9];
    double nabla_normB[3];
    double cross_prod[3];

    double r_grid_size = (rrange_arr[1] - rrange_arr[0]) / (rrange_arr[2]-1);
    double phi_grid_size = 2*M_PI / phirange_arr[2];
    double z_grid_size = (zrange_arr[1] - zrange_arr[0]) / (zrange_arr[2]-1);
    

    double x = state[0];
    double y = state[1];
    double z = state[2];
    double v_par = state[3];
    // double v_perp = state[4];

    // std::cout << "load v_par " << v_par;

    // magnetic field quad points are in cylindrical coordinates
    double r = sqrt(x*x + y*y);
    double phi = atan2(y, x); 
    
    // keep phi positive
    phi += (2*M_PI)*(phi < 0);
    
    // index into mesh to obtain nearby points
    // get correct "meta grid" for continuity
    int i = 4*((int) ((r - rrange_arr[0]) / r_grid_size) / 4);
    int j = 4*((int) ((z - zrange_arr[0]) / z_grid_size) / 4);
    int k = 4*((int) (phi / phi_grid_size) / 4);

    // std::cout << "indices: " << i << "\t" << r << "\t" << rrange_arr[0] << "\t" << r_grid_size << "\n";
    // std::cout << "position: " << x << "\t" << y << "\t" << z <<"\n";

    // normalized positions in local grid wrt e.g. r at index i
    int nr = rrange_arr[2];
    int nphi = phirange_arr[2];
    int nz = zrange_arr[2];
    double r_rel = (r -  (rrange_arr[0] + i*r_grid_size)) / r_grid_size;
    double z_rel = (z -  (zrange_arr[0] + j*z_grid_size)) / z_grid_size;
    double phi_rel = (phi - (k*phi_grid_size)) / phi_grid_size;


    // std::cout << r << "\t" << -1*(r_rel*r_grid_size - r) << "\t" << r_grid_size << "\n";
    // std::cout << z << "\t" << -1*(z_rel*z_grid_size - z) << "\t" << z_grid_size << "\n";
    // std::cout << phi << "\t" << -1*(phi_rel*phi_grid_size - phi) << "\t" << phi_grid_size << "\n";
    // std::cout << i << "\t" << j << "\t" << k << "\n";
    // std::cout << "using index " <<  (i*nz*nphi + j*nphi + k) << "\n";
    // std::cout << quadpts_arr[4*(i*nz*nphi + j*nphi + k) + 3] << "\n";
    // // std::cout << "grid point found \n";

    // // std::cout << "r_rel " << r_rel << "\t" << z_rel << "\t" << phi_rel << "\n";

    shape(r_rel, r_shape);
    shape(z_rel, z_shape);
    shape(phi_rel, phi_shape);


    // // std::cout <<"shape set \n";
    // accumulate interpolation of B
    B[0] = 0.0;
    B[1] = 0.0;            
    B[2] = 0.0;

    // interpolate the distance to the surface
    double surface_dist = 0.0;

    // // std::cout << "starting B accumulation\n";
    // quad pts are indexed r z phi
    bool is_lost = false;
    for(int ii=0; ii<=3; ++ii){             
        for(int jj=0; jj<=3; ++jj){                 
            for(int kk=0; kk<=3; ++kk){
                int wrap_k = ((k+kk) % nphi);
                if ((i+ii >= 0 & i+ii < nr) & (j+jj >= 0 & j+jj < nz)){
                    int start = 4*((i+ii)*nz*nphi + (j+jj)*nphi + (wrap_k));
                    // // std::cout << "start=" << start << "\t" << 4*nr*nz*nphi << "\n";
                    B[0] += quadpts_arr[start]   * r_shape[ii]*z_shape[jj]*phi_shape[kk];
                    B[1] += quadpts_arr[start+1] * r_shape[ii]*z_shape[jj]*phi_shape[kk];
                    B[2] += quadpts_arr[start+2] * r_shape[ii]*z_shape[jj]*phi_shape[kk];

                    is_lost = is_lost || (quadpts_arr[start+3] < 0); 
                    // // std::cout << ii << "\t" << jj << "\t" << kk << "\n";
                    // // std::cout << "interp surface dist val: " << quadpts_arr[start+3] << "\n";
                    surface_dist += quadpts_arr[start+3] * r_shape[ii]*z_shape[jj]*phi_shape[kk];
                } else{
                    // // std::cout << "bad grid index for" << r << "\t" << phi << "\t" << z <<"\n"; 
                }

            }
        }
    }

    // std::cout << "k " << k << "\t" << nphi << "\n";


    // std::cout << "is quad pt lost: " << is_lost << "\n";
    if(!is_lost){ // can't lose a particle if no quad pts are lost
        surface_dist = 1.0;    
    }
    // // std::cout << "B interpolated \n";

    // // std::cout << "r=" << r << "\t" << x << "\t" << y << "\t" << p.v_par << "\t" << surface_dist << "\n";

    // // std::cout << "particle not lost \n";

    //  Interpolate grad B: columns are partial deriv wrt r, z, phi, rows are entries of B
    //  row major order
    for(int ii=0; ii<9; ++ii){
        grad_B[ii] = 0.0;
    }
    dshape(r_rel, r_grid_size, r_dshape);
    dshape(phi_rel, phi_grid_size, phi_dshape);
    dshape(z_rel, z_grid_size, z_dshape);

    for(int ii=0; ii<=3; ++ii){             
        for(int jj=0; jj<=3; ++jj){                 
            for(int kk=0; kk<=3; ++kk){
                int wrap_k = ((k+kk) % nphi);
                if ((i+ii >= 0 & i+ii < nr) & (j+jj >= 0 & j+jj < nz)){
                    int start = 4*((i+ii)*nz*nphi + (j+jj)*nphi + (wrap_k));
                    // interpolate gradient for each entry of B, filling in each row of the gradient
                    for(int l=0; l<3; ++l){
                        double Bval = quadpts_arr[start+l];
                        grad_B[3*l]   += Bval * r_dshape[ii]*z_shape[jj]*phi_shape[kk];
                        grad_B[3*l+1] += Bval * r_shape[ii]*z_dshape[jj]*phi_shape[kk];
                        grad_B[3*l+2] += Bval * r_shape[ii]*z_shape[jj]*phi_dshape[kk];
                    }
                }

            }
        }
    }

    // // std::cout << "grad B interpolated \n";


    // convert gradient from cylindrical (r, z, phi) to cartesian coordinates (x, y, z)
    double c = cos(phi);
    double s = sin(phi);


    for(int l=0; l<3; ++l){ // iter over row
        double dfdr = grad_B[3*l];
        double dfdphi_divr = grad_B[3*l+2] / r;
        
        grad_B[3*l]   = c*dfdr - s*dfdphi_divr;
        grad_B[3*l+2] = grad_B[3*l+1]; // z index changes
        grad_B[3*l+1] = s*dfdr + c*dfdphi_divr;
    }

    // std::cout << "B" << B[0] << "\t" << B[1] << "\t" << B[2] << "\n";
    // std::cout << "grad_B" << grad_B[0] << "\t" << grad_B[1] << "\t" << grad_B[2] << "\n";
    // now compute derivatives

    // // std::cout << "starting updates \n";

    double normB = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);


    // compute \nabla |B|
    //  \nabla |B| = (\nabla B  B) / (2 |B|)
    nabla_normB[0] = (grad_B[0]*B[0] + grad_B[1]*B[1] + grad_B[2]*B[2]) / (normB);
    nabla_normB[1] = (grad_B[3]*B[0] + grad_B[4]*B[1] + grad_B[5]*B[2]) / (normB);
    nabla_normB[2] = (grad_B[6]*B[0] + grad_B[7]*B[1] + grad_B[8]*B[2]) / (normB);

    // compute B \times \nabla |B|
    cross_prod[0] = B[1]*nabla_normB[2] - B[2]*nabla_normB[1];
    cross_prod[1] = B[2]*nabla_normB[0] - B[0]*nabla_normB[2];
    cross_prod[2] = B[0]*nabla_normB[1] - B[1]*nabla_normB[0];

    // std::cout << "compute x deriv: " << v_par << "\t" << B[0] << "\t" << normB << "\t" <<  v_par << "\t" << cross_prod[0] << "\t" << m << "\t" << q << "\n";

    double v_perp2 = 2*mu*normB;

    // std::cout << "should be 0: " << (0.5*v_perp2 + pow(v_par, 2))*cross_prod[0] * m/(q*pow(normB, 3)) << "\n";
    // std::cout << "should be positive " << v_par * B[0]/normB << "\n";
    // std::cout << "v_par" << v_par << "\n";

    out[0] = v_par * B[0]/normB + (0.5*v_perp2 + pow(v_par, 2))*cross_prod[0] * m/(q*pow(normB, 3));
    out[1] = v_par * B[1]/normB + (0.5*v_perp2 + pow(v_par, 2))*cross_prod[1] * m/(q*pow(normB, 3));
    out[2] = v_par * B[2]/normB + (0.5*v_perp2 + pow(v_par, 2))*cross_prod[2] * m/(q*pow(normB, 3));

    double BdotNablaNormB = B[0]*nabla_normB[0] + B[1]*nabla_normB[1] + B[2]*nabla_normB[2];
    out[3] = -mu*BdotNablaNormB/normB;
    out[4] = normB;
    out[5] = surface_dist;

}


void trace_particle(particle_t& p, double* rrange_arr, double* zrange_arr, double* phirange_arr, double* quadpts_arr,
                        double dt, double tmax, double m, double q){
    double mu;
    int nsteps = (int) (tmax / dt);
    double surface_dist;
    // // std::cout << tmax << "\t" << dt << "\t" << nsteps << "\n";
    // double r_shape[4];
    // double phi_shape[4];
    // double z_shape[4];

    // double r_dshape[4];
    // double phi_dshape[4];
    // double z_dshape[4];

    // double B[3];
    // double grad_B[9];
    // double nabla_normB[3];
    // double cross_prod[3];

    // double r_grid_size = (rrange_arr[1] - rrange_arr[0]) / (rrange_arr[2]-1);
    // double phi_grid_size = 2*M_PI / phirange_arr[2];
    // double z_grid_size = (zrange_arr[1] - zrange_arr[0]) / (zrange_arr[2]-1);
    

    double t = 0.0;
    // for(int time_step=0; time_step<nsteps; ++time_step){

    double state[4];
    state[0] = p.x;
    state[1] = p.y;
    state[2] = p.z;
    state[3] = p.v_par;
    // state[4] = p.v_perp;

    double derivs[5];

    // dummy call to get norm B
    calc_derivs(state, derivs, rrange_arr, zrange_arr, phirange_arr, quadpts_arr, m, q, -1);
    mu = p.v_perp*p.v_perp/(2*derivs[4]);

    double k2_state[4];
    double k3_state[4];
    double k4_state[4];

    // won't use the surface distance element
    double k2[5];
    double k3[5];
    double k4[5];

    
    while(t < tmax){
        std::cout << "position: " << p.x << "\t" << p.y << "\t" << p.z << "\t" << "t=" << t  << "\n";

        // std::cout << "Time: " << t << "\n";
        /*
        * Time step ODE
        * runge-kutta 4 (see https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html)
        */

        // compute k1
        state[0] = p.x;
        state[1] = p.y;
        state[2] = p.z;
        state[3] = p.v_par;

        calc_derivs(state, derivs, rrange_arr, zrange_arr, phirange_arr, quadpts_arr, m, q, mu);

        // stop if particle lost
        surface_dist = derivs[5];
        if(surface_dist <= 0){
            std::cout << "particle lost: " << surface_dist << "\t" << t << "\t" << dt << "\n";
            p.has_left = true;
            return;
        }

        for(int i=0; i<4; ++i){
            k2_state[i] = state[i] + derivs[i]*dt/2;
        }
        calc_derivs(k2_state, k2, rrange_arr, zrange_arr, phirange_arr, quadpts_arr, m, q, mu);

        for(int i=0; i<4; ++i){
            k3_state[i] = state[i] + k2[i]*dt/2;
        }
        calc_derivs(k3_state, k3, rrange_arr, zrange_arr, phirange_arr, quadpts_arr, m, q, mu);

        for(int i=0; i<4; ++i){
            k4_state[i] = state[i] + k3[i]*dt;
        }
        calc_derivs(k4_state, k4, rrange_arr, zrange_arr, phirange_arr, quadpts_arr, m, q, mu);

        // update
        p.x +=     dt*(derivs[0] + 2*k2[0] + 2*k3[0] + k4[0])/6;
        p.y +=     dt*(derivs[1] + 2*k2[1] + 2*k3[1] + k4[1])/6;
        p.z +=     dt*(derivs[2] + 2*k2[2] + 2*k3[2] + k4[2])/6;
        p.v_par += dt*(derivs[3] + 2*k2[3] + 2*k3[3] + k4[3])/6;

        // // update
        // // // std::cout << "x update: " << p.x << "\t" <<  derivs[0] <<  "\t" << derivs[0] * dt << "\n";
        // p.x += derivs[0] * dt;
        // p.y += derivs[1] * dt;
        // p.z += derivs[2] * dt;
        // p.v_par += derivs[3] * dt;

        t += dt;
        // // std::cout << "updates complete \n";

    }
    return;
}



extern "C" vector<bool> gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> rrange,
        py::array_t<double> phirange, py::array_t<double> zrange, py::array_t<double> xyz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, bool vacuum, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria, int nparticles){

    vector<vector<array<double, 5>>> res_all(nparticles);
    vector<vector<array<double, 6>>> res_phi_hits_all(nparticles);


    //  read data in from python
    py::buffer_info xyz_buf = xyz_init.request();
    double* xyz_init_arr = static_cast<double*>(xyz_buf.ptr);
    
    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr);

    // contsins b field and then curve distance
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info r_buf = rrange.request();
    double* rrange_arr = static_cast<double*>(r_buf.ptr);

    py::buffer_info phi_buf = phirange.request();
    double* phirange_arr = static_cast<double*>(phi_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);


    particle_t particles[nparticles];
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;
        particles[i].x = xyz_init_arr[start];
        particles[i].y = xyz_init_arr[start+1];
        particles[i].z = xyz_init_arr[start+2];
        particles[i].v_par = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  particles[i].v_par* particles[i].v_par);
        particles[i].has_left = false;
        
    }

    // // std::cout << "particles initialized \n";

    double dt = 1e-12;//  1e-4*0.5*M_PI/vtotal;
    for(int p=0; p<nparticles; ++p){
        // std::cout << "tracing particle " << p << "\n";
        trace_particle(particles[p], rrange_arr, zrange_arr, phirange_arr, quadpts_arr, dt, tmax, m, q);
    }

    vector<bool> particle_loss(nparticles);
    for(int i=0; i<nparticles; ++i){
        particle_loss[i] = particles[i].has_left;
    }
    return particle_loss;
}



