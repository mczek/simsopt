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
#include <fmt/core.h>

// #include <Eigen/Core>

#include "magneticfield.h"
#include "boozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"

// #define dt 1e-7

// Particle Data Structure
typedef struct particle_t {
    double y1;  // Position Y1
    double y2;  // Position Y2
    double z;  // Position Zeta
    double v_par; // Velocity parallel
    double v_perp; // Velocity perpendicular
    double dotx;
    double doty;
    double dotz;
    double dotv_par;
    bool has_left;
} particle_t;

typedef struct workspace_t {
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
} workspace_t;

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
        // // // std::cout << c[i] <<"\n";
    }

    cudaMemcpy(c, d_c, size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}





__host__ __device__ void shape(double x, double* shape){
    shape[0] = (1.0-x)*(2.0-x)*(3.0-x)/6.0;
    shape[1] = x*(2.0-x)*(3.0-x)/2.0;
    shape[2] = x*(x-1.0)*(3.0-x)/2.0;
    shape[3] = x*(x-1.0)*(x-2.0)/6.0;
    return;         
}

__host__ __device__ void dshape(double x, double h, double* dshape){
    dshape[0] = (-(2.0-x)*(3.0-x)-(1.0-x)*(3.0-x)-(1.0-x)*(2.0-x))/(h*6.0);
    dshape[1] = ( (2.0-x)*(3.0-x)-x*(3.0-x)-x*(2.0-x))/(h*2.0);
    dshape[2] = ( (x-1.0)*(3.0-x)+x*(3.0-x)-x*(x-1.0))/(h*2.0);
    dshape[3] = ( (x-1.0)*(x-2.0)+x*(x-2.0)+x*(x-1.0))/(h*6.0);
    return;         
}

// out contains derivatives for x , y, z, v_par, and then norm of B and surface distance interpolation
__host__ __device__ void calc_derivs(double* state, double* out, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr, double m, double q, double mu, double psi0){
    /*
    * Returns     
    out[0] = ds/dtime
    out[1] = dtheta/dtime
    out[2] = dzeta/dtime

    out[3] = dvpar/dtime;
    out[4] = modB;
    

    */

    int ns = srange_arr[2];
    int nt = trange_arr[2];
    int nz = zrange_arr[2];

    // std::cout << "calling derivs with mu=\t" << mu << "\n";

    // Need to interpolate modB, modB derivs, G, and iota

    // arrays to hold weights for interpolation    
    double s_shape[4];
    double t_shape[4];
    double z_shape[4];

    /*
    * index into the grid and calculate weights
    */ 
    double s_grid_size = srange_arr[1] / (srange_arr[2]-1);
    double theta_grid_size = 2*M_PI / trange_arr[2];
    double zeta_grid_size = 2*M_PI / zrange_arr[2];
    

    // Get Boozer coordinates of current position
    double s = sqrt(state[0]*state[0] + state[1]*state[1]);
    double theta = atan2(state[1], state[0]);
    double zeta = state[2];
    double v_par = state[3];

    // map theta and zeta to [0, 2pi]
    theta = fmod(theta, 2*M_PI);
    theta += (2*M_PI)*(theta < 0);

    zeta = fmod(zeta, 2*M_PI);
    zeta += (2*M_PI)*(zeta < 0);


    // std::cout << "s,t,z=" << s << "\t" << theta << "\t" << zeta << std::endl;

    


    // index into mesh to obtain nearby points
    // get correct "meta grid" for continuity
    // keeping stz order



    int i = 3*((int) (s / s_grid_size) / 3);
    int j = 3*((int) (theta / theta_grid_size) / 3);
    int k = 3*((int) (zeta / zeta_grid_size) / 3);

    // std::cout << "i,j,k=" << i << "\t" << j << "\t" << k << std::endl;

    // // use nearest grid pts when s>1
    // if(i >= ns){
    //     std::cout << "s=" << s << std::endl;
    // }
    i = min(ns-1, i);


    // std::cout << "i,j,k=" << i << "\t" << j << "\t" << k << "\n";

    // normalized positions in local grid wrt e.g. r at index i
    // maps the position to [0,3] in the "meta grid"

    double s_rel = (s -  i*s_grid_size) / s_grid_size;
    double theta_rel = (theta -  j*theta_grid_size) / theta_grid_size;
    double zeta_rel = (zeta - k*zeta_grid_size) / zeta_grid_size;
    // std::cout << "s_rel,theta_rel,zeta_rel=" << s_rel << "\t" << theta_rel << "\t" << zeta_rel << std::endl;

    // fill shape vectors
    // this isn't particularly efficient
    shape(s_rel, s_shape);
    shape(theta_rel, t_shape);
    shape(zeta_rel, z_shape);

    /*
    From here it remains to perform the necessary interpolations
    As opposed to Cartesian coordinates, we don't need to monitor the surface dist via interpolation
    We also don't need to calculate the derivative of any of the interpolations
    This lets us interpolate everything in one set of nested loops 
    */


 


    // store interpolants in a common array, indexed the same as the columns of the quad info
    // modB, derivs of modB, G, iota
    double interpolants[6] = {0};

    // std::cout << "interpolating" << std::endl;

    // // quad pts are indexed s t z
    for(int ii=0; ii<=3; ++ii){ // s grid
        if((i+ii) < ns){
            for(int jj=0; jj<=3; ++jj){ // theta grid           
                int wrap_j = (j+jj) % nt;
                for(int kk=0; kk<=3; ++kk){ // zeta grid
                    int wrap_k = (k+kk) % nz;
                    int row_idx = (i+ii)*nt*nz + wrap_j*nz + wrap_k;
                    
                    double shape_val = s_shape[ii]*t_shape[jj]*z_shape[kk];
                    // std::cout << "modB interpolant: " << quadpts_arr[6*row_idx] << std::endl;
                    for(int zz=0; zz<6; ++zz){
                        // // std::cout << "accessing elt " << 6*row_idx + zz << "\n";
                        interpolants[zz] += quadpts_arr[6*row_idx + zz]*shape_val;
                        // if(zz == 0){
                        //     // std::cout << quadpts_arr[6*row_idx + zz] << "\n";
                        // }
                        
                    }
                    // std::cout << "running modB interpolant: " << interpolants[0] << std::endl;

                }
            }
        }

    }

    // for(int ii=0; ii<6; ++ii){
    //     std::cout << interpolants[ii] << "\n";
    // }
    


    double fak1 = m*v_par*v_par/interpolants[0] + m*mu;
    double sdot = -interpolants[2]*fak1 / (q*psi0);
    double tdot = interpolants[1]*fak1 / (q*psi0) + interpolants[5]*v_par*interpolants[0]/interpolants[4];


    out[0] = sdot*cos(theta) - s*sin(theta)*tdot;
    out[1] = sdot*sin(theta) + s*cos(theta)*tdot;
    out[2] = v_par*interpolants[0]/interpolants[4];
    out[3] = -(interpolants[5]*interpolants[2] + interpolants[3])*mu*interpolants[0] / interpolants[4];

   
    out[4] = interpolants[0];
    

}


__host__ __device__  void trace_particle(particle_t& p, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double dt, double tmax, double m, double q, double psi0){

    double mu;
    double t = 0.0;

    double state[4];
    state[0] = p.y1;
    state[1] = p.y2;
    state[2] = p.z;
    state[3] = p.v_par;
    // state[4] = p.v_perp;

    double derivs[6];

    // dummy call to get norm B
    // std::cout << "dummy call to calc_derivs \n";
    calc_derivs(state, derivs, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, -1, psi0);
    mu = p.v_perp*p.v_perp/(2*derivs[4]);

    // std::cout << "initial modB " << derivs[4] << std::endl; 

    // std::cout << "modB interp " << derivs[4] << std::endl;
    const double a21 = 1.0 / 5.0;
    const double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
    const double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
    const double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0, a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
    const double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0, a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
    const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    // const double bhat1 = 5179.0 / 57600.0, bhat3 = 7571.0 / 16695.0, bhat4 = 393.0 / 640.0, bhat5 = -92097.0 / 339200.0, bhat6 = 187.0 / 2100.0, bhat7 = 1.0 / 40.0;
    const double bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;


    double k2[6], k3[6], k4[6], k5[6], k6[6], k7[6];
    double x_temp[4], x_new[4], x_err[4];


    int counter = 0;
    while(t < tmax){
        // if(counter % 10 == 0){
        // std::cout << "position: " << p.y1 << "\t" << p.y2 << "\t" << p.z << "\t" << "t=" << t  << "\t dt= " << dt << std::endl;
        // }

        // // std::cout << "Time: " << t << "\n";
        /*
        * Time step ODE
        * runge-kutta 4 (see https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html)
        * 
        * Adaptive Dopri5 time step: p.167
        * https://link.springer.com/book/10.1007/978-3-540-78862-1
        */

        // compute k1
        state[0] = p.y1;
        state[1] = p.y2;
        state[2] = p.z;
        state[3] = p.v_par;

        calc_derivs(state, derivs, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, mu, psi0);
        // std::cout << "k1 " << derivs[0] << "\t" << derivs[1] << "\t" << derivs[2] << "\t" << derivs[3] << "\n";
        // stop if particle lost
 
        
        // Compute k2
        for (int i = 0; i < 4; i++) x_temp[i] = state[i] + dt * a21 * derivs[i];
        calc_derivs(x_temp, k2, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, mu, psi0);
        // std::cout << "k2 " << k2[0] << "\t" << k2[1] << "\t" << k2[2] << "\t" << k2[3] << "\n";

        // Compute k3
        for (int i = 0; i < 4; i++) x_temp[i] = state[i] + dt * (a31 * derivs[i] + a32 * k2[i]);
        calc_derivs(x_temp, k3, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, mu, psi0);
        // std::cout << "k3 " << k3[0] << "\t" << k3[1] << "\t" << k3[2] << "\t" << k3[3] << "\n";

        // Compute k4
        for (int i = 0; i < 4; i++) x_temp[i] = state[i] + dt * (a41 * derivs[i] + a42 * k2[i] + a43 * k3[i]);
        calc_derivs(x_temp, k4, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, mu, psi0);

        // Compute k5
        for (int i = 0; i < 4; i++) x_temp[i] = state[i] + dt * (a51 * derivs[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        calc_derivs(x_temp, k5, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, mu, psi0);

        // Compute k6
        for (int i = 0; i < 4; i++) x_temp[i] = state[i] + dt * (a61 * derivs[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        calc_derivs(x_temp, k6, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, mu, psi0);

        // Compute new state
        for (int i = 0; i < 4; i++) {
            x_new[i] = state[i] + dt * (b1 * derivs[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
        }

        // Compute k7 for error estimation
        calc_derivs(x_new, k7, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, mu, psi0);
        
        // Compute  error
        // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
        double tol=1e-9;
        // // std::cout << "error elts \n";
        double err = 0;
        bool accept = true;
        for (int i = 0; i < 4; i++) {
            x_err[i] = dt*(bhat1 * derivs[i] + bhat3 * k3[i] + bhat4 * k4[i] + bhat5 * k5[i] + bhat6 * k6[i] + bhat7 * k7[i]);
            x_err[i] = fabs(x_err[i]) / (tol + tol*(fabs(state[i]) + fabs(derivs[i])));      
            // // std::cout << std::abs(x_err[i]) << "\n";
            err = fmax(err, x_err[i]);
        }

        // // std::cout << "err= " << err << "\n";

        // Compute new step size

        // // std::cout << "intermediate val=" << 0.9*pow(err, -1.0/5.0) << "\n";
        double dt_new = dt*0.9*pow(err, -1.0/5.0);
        dt_new = max(dt_new, 0.2 * dt);  // Limit step size reduction
        dt_new = min(dt_new, 5.0 * dt);  // Limit step size increase
        if ((0.5 < err) & (err < 1.0)){
            dt_new = dt;
        }
        // dt_new = std::max(dt_new, 1e-9); // Limit smallest step size
        // // std::cout << "dt_new= " << dt_new << "\t dt=" << dt << "\n";
        if (err <= 1.0) {
            // // std::cout << "point accepted\n";
            // Accept the step
            t += dt;
            dt = min(dt_new, tmax - t);

            p.y1 = x_new[0];
            p.y2 = x_new[1];
            p.z = x_new[2];
            p.z = fmod(p.z, 2*M_PI);
            p.z += (2*M_PI)*(p.z < 0);
            p.v_par = x_new[3];
        } else {
            // Reject the step and try again with smaller dt
            dt = dt_new;
        }

        double s = sqrt(p.y1*p.y1 + p.y2*p.y2);
        if(s >= 1){
            // // std::cout << "particle lost: " << surface_dist << "\t" << t << "\t" << dt << "\n";
            p.has_left = true;
            return;
        }

        counter++;

    }
    return;
}

__global__ void particle_trace_kernel(particle_t* particles, double* rrange_arr, double* zrange_arr, double* phirange_arr, double* quadpts_arr,
                        double dt, double tmax, double m, double q, double psi0, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        trace_particle(particles[idx], rrange_arr, zrange_arr, phirange_arr, quadpts_arr, dt, tmax, m, q, psi0);
    }
}

extern "C" vector<bool> gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, double psi0, int nparticles){

    // vector<vector<array<double, 5>>> res_all(nparticles);
    // vector<vector<array<double, 6>>> res_phi_hits_all(nparticles);

    // std::cout << "calling gpu tracing\n";

    //  read data in from python
    auto ptr = stz_init.data();
    int size = stz_init.size();
    double stz_init_arr[size];
    std::memcpy(stz_init_arr, ptr, size * sizeof(double));

    // py::buffer_info xyz_buf = xyz_init.request();
    // double* xyz_init_arr = static_cast<double*>(xyz_buf.ptr);
    
    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr);

    // contsins b field and then curve distance
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);


    particle_t* particles =  new particle_t[nparticles];

    // convert to alternative coordinates
    /*
    * y1 = s*cos(theta)
    * y2 = s*sin(theta)
    */

    // std::cout << "loading particles" << "\n";

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double s = stz_init_arr[start];
        double theta = stz_init_arr[start+1];
        
        // convert to alternative coordinates
        particles[i].y1 = s*cos(theta);
        particles[i].y2 = s*sin(theta);
        
        particles[i].z = stz_init_arr[start+2];
        particles[i].v_par = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  particles[i].v_par* particles[i].v_par);
        particles[i].has_left = false;
        
    }

   

    double dt = 1e-5*0.5*M_PI/vtotal;
    // for(int p=0; p<nparticles; ++p){
    //     // std::cout << "tracing particle " << p << "\n";
    //     trace_particle(particles[p], srange_arr, trange_arr, zrange_arr, quadpts_arr, dt, tmax, m, q, psi0);
    // }

    
    particle_t* particles_d;
    cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t));
    cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);

    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);


    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    int nthreads = 256;
    int nblks = nparticles / nthreads + 1;
    particle_trace_kernel<<<nblks, nthreads>>>(particles_d, srange_d, zrange_d, trange_d, quadpts_d, dt, tmax, m, q, psi0, nparticles);

    cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);

    
    vector<bool> particle_loss(nparticles);
    for(int i=0; i<nparticles; ++i){
        particle_loss[i] = particles[i].has_left;
    }

    // delete[] workspaces;
    delete[] particles;

    return particle_loss;
}



