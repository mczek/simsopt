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

#define BATCH_SIZE 1000
#define PARTICLES_PER_BLOCK 128

// Particle Data Structure
typedef struct particle_t {
    double state[4];
    double v_perp; // Velocity perpendicular
    double v_total;
    bool has_left;
    double dt;
    double dtmax;
    double t;
    double mu;
    double derivs[42] = {0.0};
    // double k2[6], k3[6], k4[6], k5[6], k6[6], k7[6];   
    double x_temp[4], x_err[4];
    double s_shape[4], t_shape[4], z_shape[4];
    int i, j, k;
    double interpolation_loc[3];
    bool symmetry_exploited;
    int id;
    int step_attempt, step_accept;
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

// __host__ __device__ void dshape(double x, double h, double* dshape){
//     dshape[0] = (-(2.0-x)*(3.0-x)-(1.0-x)*(3.0-x)-(1.0-x)*(2.0-x))/(h*6.0);
//     dshape[1] = ( (2.0-x)*(3.0-x)-x*(3.0-x)-x*(2.0-x))/(h*2.0);
//     dshape[2] = ( (x-1.0)*(3.0-x)+x*(3.0-x)-x*(x-1.0))/(h*2.0);
//     dshape[3] = ( (x-1.0)*(x-2.0)+x*(x-2.0)+x*(x-1.0))/(h*6.0);
//     return;         
// }

__host__  __device__ __forceinline__ void interpolate(particle_t& p, const double* __restrict__ data, double* out, const double* __restrict__ srange_arr, const double* __restrict__ trange_arr, const double* __restrict__ zrange_arr, int n){


    int ns = srange_arr[2];
    int nt = trange_arr[2];
    int nz = zrange_arr[2];

    // Need to interpolate modB, modB derivs, G, and iota

    /*
    From here it remains to perform the necessary interpolations
    As opposed to Cartesian coordinates, we don't need to monitor the surface dist via interpolation
    We also don't need to calculate the derivative of any of the interpolations
    This lets us interpolate everything in one set of nested loops 
    */

    // store interpolants in a common array, indexed the same as the columns of the quad info
    // modB, derivs of modB, G, iota
    // double interpolants[6] = {0};

    double thread_total = 0.0;
    // // quad pts are indexed s t z
    for(int ii=0; ii<=3; ++ii){ // s grid
        if((p.i+ii) < ns){
            for(int jj=0; jj<=3; ++jj){ // theta grid           
                int wrap_j = (p.j+jj) % nt;
                for(int kk=0; kk<=3; ++kk){ // zeta grid
                    int wrap_k = (p.k+kk) % nz;
                    int row_idx = (p.i+ii)*nt*nz + wrap_j*nz + wrap_k;
                    
                    double shape_val = p.s_shape[ii]*p.t_shape[jj]*p.z_shape[kk];
                    // std::cout << row_idx << " modB interpolant: " << data[n*row_idx] << std::endl;

                    // fmt::print("modB val={}, s_shape={}, t_shape={}, z_shape={}\n", data[n*row_idx], s_shape[ii], t_shape[jj], z_shape[kk]);
                        // // std::cout << "accessing elt " << 6*row_idx + zz << "\n";
                    for(int zz=0; zz<6; ++zz){
                        out[zz] += data[n*row_idx + zz]*shape_val;
                    }
                        // if(zz == 0){
                        //     // std::cout << quadpts_arr[6*row_idx + zz] << "\n";
                        // }
                        

                    // std::cout << "running modB interpolant: " << interpolants[0] << std::endl;

                }
            }
        }

    }

}

// out contains derivatives for x , y, z, v_par, and then norm of B and surface distance interpolation
__host__  __device__ void calc_derivs(particle_t& p, double* out, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr, double m, double q, double mu, double psi0){
    /*
    * Returns     
    out[0] = ds/dtime
    out[1] = dtheta/dtime
    out[2] = dzeta/dtime

    out[3] = dvpar/dtime;
    out[4] = modB;
    

    */
    

    // double* loc = loc_shared + 3* block_part_id;
    double interpolants[6] = {0.0};
    

    
   
    // printf("interpolation loc: %.15e, %.15e, %.15e\n", p.interpolation_loc[0], p.interpolation_loc[1], p.interpolation_loc[2]);

    interpolate(p, quadpts_arr, interpolants, srange_arr, trange_arr, zrange_arr, 6);

    // printf("interpolants:  %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", interpolants[0], interpolants[1], interpolants[2], interpolants[3], interpolants[4], interpolants[5]);

    
    double s = sqrt(p.x_temp[0]*p.x_temp[0] + p.x_temp[1]*p.x_temp[1]);
    double theta = atan2(p.x_temp[1], p.x_temp[0]);
    double z = p.x_temp[2];
    double v_par = p.x_temp[3];
    if(p.symmetry_exploited){
        interpolants[2] *= -1.0;
        interpolants[3] *= -1.0;
    }
    // printf("s=%.15e, theta=%.15e, zeta=%.15e, vpar=%.15e\n", s, theta, z, v_par);
    // printf("interpolants:  %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", interpolants[0], interpolants[1], interpolants[2], interpolants[3], interpolants[4], interpolants[5]);

    // printf("modB=%.15e, modB_derivs=%.15e, %.15e, %.15e, G=%.15e, iota=%.15e\n", interpolants[0], interpolants[1], interpolants[2], interpolants[3], interpolants[4], interpolants[5]);
    // fmt::print("modB ={}, modB derivs={} {} {}, G={}, iota={}\n", interpolants[0], interpolants[1], interpolants[2], interpolants[3], interpolants[4], interpolants[5]);
    // std::cout << "\n";

    // fmt::print("m={}, v_par={}, mu={}\n", m, v_par, mu);
    // printf("m=%.15e, mu=%.15e, psi0=%.15e, q=%.15e\n", m, mu, psi0, q);

    double fak1 = m*v_par*v_par/interpolants[0] + m*mu;
    double sdot = -interpolants[2]*fak1 / (q*psi0);
    double tdot = interpolants[1]*fak1 / (q*psi0) + interpolants[5]*v_par*interpolants[0]/interpolants[4];

    // fmt::print("fak1={}, sdot={}, tdot={}\n", fak1, sdot, tdot);

    // printf("fak1=%.15e, sdot=%.15e, tdot=%.15e\n", fak1, sdot, tdot);

    out[0] = sdot*cos(theta) - s*sin(theta)*tdot;
    out[1] = sdot*sin(theta) + s*cos(theta)*tdot;
    out[2] = v_par*interpolants[0]/interpolants[4];
    out[3] = -(interpolants[5]*interpolants[2] + interpolants[3])*mu*interpolants[0] / interpolants[4];

    // fmt::print("derivs = {} {} {} {}\n\n", out[0], out[1], out[2], out[3]);
    out[4] = interpolants[0]; // modB
    out[5] = interpolants[4]; // G

    // printf("calc_derivs out vals: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", out[0], out[1], out[2], out[3], out[4], out[5]);

}



__host__ __device__ void build_state(particle_t& p, int deriv_id, double* srange_arr, double* trange_arr, double* zrange_arr){
   

    // const double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0, a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
    const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    // const double bhat1 = 5179.0 / 57600.0, bhat3 = 7571.0 / 16695.0, bhat4 = 393.0 / 640.0, bhat5 = -92097.0 / 339200.0, bhat6 = 187.0 / 2100.0, bhat7 = 1.0 / 40.0;
    // const double bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;

    double wgts[6] = {0.0}; 
    // printf("state=%.15e, %.15e, %.15e\n", p.state[0], p.state[1], p.state[2]);

    for (int i = 0; i < 4; i++) {
        p.x_temp[i] = p.state[i];
    }
    // printf("xtemp=%.15e, %.15e, %.15e\n", p.x_temp[0], p.x_temp[1], p.x_temp[2]);

    switch(deriv_id){
        case 0:
            // wgts = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            break;
        case 1:
            // wgts = {1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            wgts[0] = 1.0/5.0;
            break;
        case 2:
            // wgts = {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0};
            wgts[0] = 3.0 / 40.0;
            wgts[1] = 9.0 / 40.0;
            break;
        case 3:
            // wgts = {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0};
            wgts[0] = 44.0 / 45.0;
            wgts[1] = -56.0 / 15.0;
            wgts[2] = 32.0 / 9.0;
            break;
        case 4:
            // wgts = {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0};
            wgts[0] = 19372.0 / 6561.0;
            wgts[1] = -25360.0 / 2187.0;
            wgts[2] = 64448.0 / 6561.0;
            wgts[3] = -212.0 / 729.0;
            break;
        case 5:
            // wgts = {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,-5103.0 / 18656.0, 0.0, 0.0};
            wgts[0] = 9017.0 / 3168.0;
            wgts[1] = -355.0 / 33.0;
            wgts[2] = 46732.0 / 5247.0;
            wgts[3] = 49.0 / 176.0;
            wgts[4] = -5103.0 / 18656.0;
            break;
        case 6:
            // wgts = {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0};
            wgts[0] = 35.0 / 384.0;
            wgts[2] = 500.0 / 1113.0;
            wgts[3] = 125.0 / 192.0; 
            wgts[4] = -2187.0 / 6784.0;
            wgts[5] = 11.0 / 84.0;
            break;
        default:
            break;
    }

    // printf("deriv pt first: %.15e, %.15e, %.15e, %.15e, %.15e\n", p.dt, p.x_temp[0], p.x_temp[1], p.x_temp[2], p.x_temp[3]);
    // printf("wgts: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", wgts[0], wgts[1], wgts[2], wgts[3], wgts[4], wgts[5]);


    for (int j=0; j<6; ++j){
        for(int i=0; i<4; ++i){
            // printf("contribution: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", p.dt, wgts[j], p.derivs[6*j+i], p.dt * wgts[j], wgts[j] * p.derivs[6*j+i], p.dt * p.derivs[6*j+i], p.dt * wgts[j] * p.derivs[6*j+i]);
            p.x_temp[i] += p.dt * wgts[j] * p.derivs[6*j+i];
        }
    } 


    // printf("deriv pt second: %.15e, %.15e, %.15e, %.15e\n", p.x_temp[0], p.x_temp[1], p.x_temp[2], p.x_temp[3]);


    double s = sqrt(p.x_temp[0]*p.x_temp[0] + p.x_temp[1]*p.x_temp[1]);
    double theta = atan2(p.x_temp[1], p.x_temp[0]);
    double z = p.x_temp[2];
    double v_par = p.x_temp[3];
    




    // fmt::print("s={}, theta={}, zeta={}, v_par={}\n", s, theta, z, v_par);

    // fmt::print("m={}, mu={}, q={}, psi0={}\n", m, mu, q, psi0);

    // exploit potential symmetry
    
    // we want to exploit periodicity in the B-field, but leave sine(theta) unchanged
    double t = fmod(theta, 2*M_PI);
    t += 2*M_PI*(t < 0);

    // we can modify z because it's only used to access the B-field location
    double period = zrange_arr[1];
    z = fmod(z, period);
    z += period*(z < 0);

    
    // exploit stellarator symmetry
    p.symmetry_exploited = t > M_PI;
    if(p.symmetry_exploited){
        z = period - z;
        t = 2*M_PI - t;
        // std::cout << "symmetry exploited\n";

    }
    p.interpolation_loc[0] = s;
    p.interpolation_loc[1] = t;
    p.interpolation_loc[2] = z;


    /*
    * index into the grid and calculate weights
    */ 
    double s_grid_size = (srange_arr[1]-srange_arr[0]) / (srange_arr[2]-1);
    double theta_grid_size = (trange_arr[1]-trange_arr[0]) / (trange_arr[2]-1);
    double zeta_grid_size = (zrange_arr[1]-zrange_arr[0]) / (zrange_arr[2]-1);

    // Get Boozer coordinates of current position
    // double s = loc[0];
    // double t = loc[1];
    // double z = loc[2];


    p.i = 3*((int) ((s - srange_arr[0]) / s_grid_size) / 3);
    p.j = 3*((int) ((t - trange_arr[0]) / theta_grid_size) / 3);
    p.k = 3*((int) ((z - zrange_arr[0]) / zeta_grid_size) / 3);

    //printf("i=%d, j=%d, k=%d\n", p.i, p.j, p.k);


    p.i = min(p.i, (int)srange_arr[2]-4);
    p.j = min(p.j, (int)trange_arr[2]-4);
    p.k = min(p.k, (int)zrange_arr[2]-4);

    // normalized positions in local grid wrt e.g. r at index i
    // maps the position to [0,3] in the "meta grid"

    double s_rel = (s -  p.i*s_grid_size - srange_arr[0]) / s_grid_size;
    double theta_rel = (t -  p.j*theta_grid_size - trange_arr[0]) / theta_grid_size;
    double zeta_rel = (z - p.k*zeta_grid_size - zrange_arr[0]) / zeta_grid_size;

    shape(s_rel, p.s_shape);
    shape(theta_rel, p.t_shape);
    shape(zeta_rel, p.z_shape);

}

// set initial time step, calculate mu
__host__ __device__ void setup_particle(particle_t& p, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                         double tmax, double m, double q, double psi0){
                             // double mu;
    p.t = 0.0;
    p.dt = 0.0;
    build_state(p, 0, srange_arr, trange_arr, zrange_arr);

    // printf("post build_state interpolation_loc: %.15e, %.15e, %.15e\n", p.interpolation_loc[0], p.interpolation_loc[1], p.interpolation_loc[2]);

    // p.state[0] = p.y1;
    // p.state[1] = p.y2;
    // p.state[2] = p.z;
    // p.state[3] = p.v_par;
    // state[4] = p.v_perp;


    // dummy call to get norm B
    // std::cout << "dummy call to calc_derivs \n";
    calc_derivs(p, p.derivs, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, -1, psi0);
    
    // printf("setting mu: %.15e, %.15e, %.15e, %.15e, %.15e\n", p.mu, p.v_perp, p.v_perp*p.v_perp, 2*p.derivs[4], p.v_perp*p.v_perp / 2*p.derivs[4]);
    // printf("derivs[4]: %.15e\n", p.derivs[4]);

    double v_perp2 = p.v_perp*p.v_perp;
    double denom = 1 / (2*p.derivs[4]);
    p.mu = v_perp2 * denom;
    // printf("derivs[4]: %.15e, %.15e, %.15e, %.15e\n", p.derivs[4], v_perp2, denom, v_perp2*denom);
    

    // printf("setting mu: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", p.mu, p.v_perp, p.v_perp*p.v_perp, p.derivs[4], p.v_perp*p.v_perp/(2*p.derivs[4]), (p.v_perp*p.v_perp)/(2*p.derivs[4]));

        // dtmax = 0.5*M_PI*G / (modB*vtotal)
    p.dtmax = 0.5*M_PI*abs(p.derivs[5]) / (p.derivs[4]*p.v_total);
    p.dt = 1e-3*p.dtmax;

    // printf("dtmax=%.15e, dt=%.15e, G=%.15e, modB=%.15e, v_total=%.15e\n", p.dtmax, p.dt, p.derivs[5], p.derivs[4], p.v_total);

}

__host__ __device__ void adjust_time(particle_t& p, double tmax){
    if(p.has_left){
        return;
    }

    const double bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;

    // Compute  error
    // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // resolve typo in boost docs: https://numerical.recipes/book.html
    double atol=1e-9;
    double rtol=1e-9;
    // // std::cout << "error elts \n";
    double err = 0.0;
    bool accept = true;
    for (int i = 0; i < 4; i++) {
        p.x_err[i] = p.dt*(bhat1 * p.derivs[i] + bhat3 * p.derivs[12+i] + bhat4 * p.derivs[18+i] + bhat5 * p.derivs[24+i] + bhat6 * p.derivs[30+i] + bhat7 * p.derivs[36+i]);
        //printf("raw error: %.15e\tnumerator: %.15e\tdenominator: %.15e\n", p.x_err[i], fabs(p.x_err[i]), (atol + rtol*(fabs(p.state[i]) + p.dt*fabs(p.derivs[i]))));
       
        if(i==3){
            atol *= 1e5;
        }
        p.x_err[i] = fabs(p.x_err[i]) / (atol + rtol*(fabs(p.state[i]) + p.dt*fabs(p.derivs[i])));      
        // // std::cout << std::abs(x_err[i]) << "\n";
        err = fmax(err, p.x_err[i]);
        //printf("running max err: %.15e\n", err);
    }
    //printf("state elements: %.15e, %.15e, %.15e, %.15e\n", p.state[0], p.state[1], p.state[2], p.state[3]);
    //printf("deriv elements: %.15e, %.15e, %.15e, %.15e\n", p.derivs[0], p.derivs[1], p.derivs[2], p.derivs[3]);
    //printf("error elements: %.15e, %.15e, %.15e, %.15e\n", p.x_err[0], p.x_err[1], p.x_err[2], p.x_err[3]);

    // // std::cout << "err= " << err << "\n";

    // Compute new step size

    // // std::cout << "intermediate val=" << 0.9*pow(err, -1.0/5.0) << "\n";
    double dt_new = p.dt*0.9*pow(err, -1.0/4.0);
    // if(err > 1.0)
    dt_new = fmax(dt_new, 0.2 * p.dt);  // Limit step size reduction
    dt_new = fmin(dt_new, 5.0 * p.dt);  // Limit step size increase
    dt_new = fmin(p.dtmax, dt_new);
    if ((0.5 < err) & (err < 1.0)){
        dt_new = p.dt;
    }
    // dt_new = std::max(dt_new, 1e-9); // Limit smallest step size
    // // std::cout << "dt_new= " << dt_new << "\t dt=" << dt << "\n";
    p.step_attempt++;
    //printf("t=%.15e, err=%.15e, position %.15e, %.15e, %.15e, %.15e, dt=%.15e, dt_new=%.15e\n", p.t, err, p.state[0], p.state[1], p.state[2], p.state[3], p.dt, dt_new);
    if (err <= 1.0) {
        // // std::cout << "point accepted\n";
        // Accept the step
        p.t += p.dt;
        p.dt = fmin(dt_new, tmax - p.t);

        p.state[0] = p.x_temp[0];
        p.state[1] = p.x_temp[1];
        p.state[2] = p.x_temp[2];
        // p.z = fmod(p.z, zrange_arr[1]);
        // p.z += zrange_arr[1]*(p.z < 0);
        p.state[3] = p.x_temp[3];

        double s = sqrt(p.state[0]*p.state[0] + p.state[1]*p.state[1]);
        p.has_left = s >= 1;
        p.step_accept++;


    } else {
        // Reject the step and try again with smaller dt
        p.dt = dt_new;
    }

}
__host__ __device__    void trace_particle(particle_t& p, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                         double tmax, double m, double q, double psi0){

   
    setup_particle(p, srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q, psi0);

    int counter = 0;


    while(p.t < tmax){
        //printf("particle %d position %.15e, %.15e, %.15e, %.15e, dt=%.15e\n", p.id, p.state[0], p.state[1], p.state[2], p.state[3], p.dt);


        // if(counter > 1000000){
        //     printf("particle %d cutoff at position %.15e, %.15e, %.15e, %.15e, dt=%.15e\n", p.id, p.state[0], p.state[1], p.state[2], p.state[3], p.dt);
        //     return;
        // }
      

        for(int k=0; k<7; ++k){
            build_state(p, k, srange_arr, trange_arr, zrange_arr);
            calc_derivs(p, p.derivs + 6*k, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, p.mu, psi0);
        }
        adjust_time(p, tmax);
        
        double s = sqrt(p.state[0]*p.state[0] + p.state[1]*p.state[1]);
        if(s >= 1){
            p.has_left = true;
            return;
        }

        counter++;

    }
    return;
}

__global__ void particle_trace_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double tmax, double m, double q, double psi0, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        trace_particle(particles[idx], srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q, psi0);
    }
}


__global__ void setup_particle_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double tmax, double m, double q, double psi0, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int particle_id = idx / 6;
    if(particle_id < nparticles){
        setup_particle(particles[particle_id], srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q, psi0);
    }
}

__global__ void build_state_kernel(particle_t* particles, int deriv_id, double* srange_arr, double* trange_arr, double* zrange_arr, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        build_state(particles[idx], deriv_id, srange_arr, trange_arr, zrange_arr);
    }
}

 
__global__ void calc_derivs_kernel(particle_t* particles, int deriv_id, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr, double m, double q, double psi0, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int particle_id = idx / 6;
    if(particle_id < nparticles){
        calc_derivs(particles[particle_id], particles[particle_id].derivs + 6*deriv_id, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, particles[particle_id].mu, psi0);
    }
}


__global__ void count_done_kernel(particle_t* particles, double tmax, int *total_done, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        int is_done = (int) (particles[idx].has_left || (particles[idx].t >= tmax));
        atomicAdd(total_done, is_done);
    }
}

__global__ void adjust_time_kernel(particle_t* particles, double tmax, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        adjust_time(particles[idx], tmax);
    }
}


extern "C" vector<double> gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
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

    // contains b field
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
        particles[i].state[0] = s*cos(theta);
        particles[i].state[1] = s*sin(theta);
        
        particles[i].state[2] = stz_init_arr[start+2];
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;
        
        // particles[i].dt = 0.0; //initialize to zero for build_state
        // ensure data is initialized
        // particles[i].derivs = {0.0};
        // particles[i].x_temp = {0.0};
        // particles[i].x_err = {0.0};
        // particles[i].s_shape = {0.0};
        // particles[i].t_shape = {0.0};
        // particles[i].z_shape = {0.0};
        // particles[i].interpolation_loc = {0.0};

    }

    // int nthreads = 256;
    // int nblks = nparticles / nthreads + 1;


    // particle_t* particles_d;
    // cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t));
    // cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);

    // double* srange_d;
    // cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    // cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    // double* zrange_d;
    // cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    // cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    // double* trange_d;
    // cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    // cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);


    // double* quadpts_d;
    // cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    // cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    // int threads_per_particle = 6;
    // int threads_per_block = threads_per_particle*PARTICLES_PER_BLOCK;
    // int interpolation_blocks = nparticles / PARTICLES_PER_BLOCK + 1;

    // setup_particle_kernel<<<interpolation_blocks, threads_per_block>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, tmax, m, q, psi0, nparticles);

    // cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);

    
    // int* total_done_d;
    // cudaMalloc((void**)&total_done_d, sizeof(int));
    // cudaMemset(total_done_d, 0, sizeof(int));
    // count_done_kernel<<<nblks, nthreads>>>(particles_d, tmax, total_done_d, nparticles);

    // int total_done;
    // cudaMemcpy(&total_done, total_done_d, sizeof(int), cudaMemcpyDeviceToHost);
    // fmt::print("number done = {}\n", total_done);
   
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    
    // particle_trace_kernel<<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, tmax, m, q, psi0, nparticles);

    // int total_done = 0;

    // while (total_done < nparticles){
    //     fmt::print("number done = {}\n", total_done);
    //      // double dt = 1e-5*0.5*M_PI/vtotal;

    //     for(int i=0; i<BATCH_SIZE; ++i){
    //         // advance 1 step
    //         for(int k=0; k<7; ++k){
    //             // cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);
    //             build_state_kernel<<<nblks, nthreads>>>(particles_d, k, srange_d, trange_d, zrange_d, nparticles); 
    //             // cudaDeviceSynchronize();
    //             // cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);


    //             // cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);
    //             calc_derivs_kernel<<<interpolation_blocks, threads_per_block>>>(particles_d, k, srange_d, trange_d, zrange_d, quadpts_d, m, q, psi0, nparticles); 
    //             // cudaDeviceSynchronize();
    //             // cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);
    //             // for(int p=0; p<nparticles; ++p){
    //             // // std::cout << "tracing particle " << p << std::endl;
 
    //             //     // build_state(particles[p], k);
    //             //     calc_derivs(particles[p].x_temp, particles[p].derivs + 6*k, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, particles[p].mu, psi0);
    //             // }
    //             // adjust_time(particles[p], tmax);




    //         }



    //         // cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);
    //         adjust_time_kernel<<<nblks, nthreads>>>(particles_d, tmax, nparticles);
    //         // cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);
    //     }

    //     // total_done = 0;
    //     // for(int i=0; i<nparticles; ++i){
    //     //     total_done += (int) (particles[i].has_left || (particles[i].t >= tmax));
    //     // }

    //     cudaMemset(total_done_d, 0, sizeof(int));
    //     count_done_kernel<<<nblks, nthreads>>>(particles_d, tmax, total_done_d, nparticles);
    //     cudaMemcpy(&total_done, total_done_d, sizeof(int), cudaMemcpyDeviceToHost);



    // }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";

   
    
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
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    particle_trace_kernel<<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, tmax, m, q, psi0, nparticles);

    cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";

    // for(int i=0; i<nparticles; ++i){
    //     std::cout << "tracing particle " << i << std::endl;
    //     trace_particle(particles[i], srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q, psi0);
    // }


    
    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        // last location in Boozer coordinates
        particle_output[7*i] = sqrt(y1*y1 + y2*y2);
        particle_output[7*i + 1] = atan2(y1, y2);
        particle_output[7*i + 2] = z;
        particle_output[7*i + 3] = v_par;
        particle_output[7*i + 4] = particles[i].t;
        particle_output[7*i + 5] = particles[i].step_accept;
        particle_output[7*i + 6] = particles[i].step_attempt;
    }


    delete[] particles;

    return particle_output;
}

extern "C" py::array_t<double> test_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, int n){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);

    double out[n];

    // double s = loc_arr[0];
    double t = loc_arr[1];
    double z = loc_arr[2];
    // we want to exploit periodicity in the B-field, but leave sine(theta) unchanged
    t = fmod(t, 2*M_PI);
    t += 2*M_PI*(t < 0);

    // we can modify z because it's only used to access the B-field location
    double period = zrange_arr[1];
    z = fmod(z, period);
    z += period*(z < 0);

    
    // exploit stellarator symmetry
    bool symmetry_exploited = t > M_PI;
    if(symmetry_exploited){
        z = period - z;
        t = 2*M_PI - t;
    }
    loc_arr[1] = t;
    loc_arr[2] = z;

    // interpolate(loc_arr, quadpts_arr, out, srange_arr, trange_arr, zrange_arr, n);


    if(symmetry_exploited){
        out[2] *= -1.0;
        out[3] *= -1.0;
    }

    auto result = py::array_t<double>(n, out);
    return result;

}

__global__ void test_gpu_interpolation_kernel(double* quad_pts, double* srange, double* trange, double* zrange, double* loc, double* out, int n, int n_points){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n_points){
        double* loc_arr = loc + 3*idx;
        double* out_arr  =  out + idx*n;

        particle_t p;
        double s = loc_arr[0];
        double t = loc_arr[1];
        double z = loc_arr[2];

        p.state[0] = s*cos(t);
        p.state[1] = s*sin(t);
        p.state[2] = z;

        p.dt = 1e-3; //needed for build_state

        // printf("s=%.15e, t=%.15e, z=%.15e\n", s, t, z);
        // printf("y1=%.15e, y2=%.15e, z=%.15e\n", p.state[0], p.state[1], p.state[2]);
        build_state(p, 0, srange, trange, zrange);
        // printf("xtemp=%.15e, %.15e, %.15e\n", p.x_temp[0], p.x_temp[1], p.x_temp[2]);

        interpolate(p, quad_pts, out_arr, srange, trange, zrange, n);

        // printf("modB=%.15e, modB_derivs=%.15e, %.15e, %.15e, G=%.15e, iota=%.15e\n", out_arr[0], out_arr[1], out_arr[2], out_arr[3], out_arr[4], out_arr[5]);


        if(p.symmetry_exploited){
            out_arr[2] *= -1.0;
            out_arr[3] *= -1.0;
        }



    }
}



extern "C" py::array_t<double> test_gpu_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, int n, int n_points){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);


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

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);


    double* out_d;
    cudaMalloc((void**)&out_d, n*n_points * sizeof(double));



    int nthreads = 256;
    int nblks = n_points / nthreads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_interpolation_kernel<<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, out_d, n, n_points);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "interpolation kernel time (ms): " << milliseconds<< "\n";
    
    double out[n*n_points];
    cudaMemcpy(&out, out_d, n*n_points * sizeof(double), cudaMemcpyDeviceToHost);
    auto result = py::array_t<double>(n*n_points, out);
    return result;

}


__global__ void test_gpu_derivs_kernel(double* quad_pts, double* srange, double* trange, double* zrange, double* loc, double* vpar, double vtotal, double* out, double m, double q, double psi0, int n_points){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n_points){
        double* loc_arr = loc + 3*idx;
        double* out_arr  =  out + 4*idx;
        double vpar_val = vpar[idx];

        particle_t p;
        double s = loc_arr[0];
        double t = loc_arr[1];
        double z = loc_arr[2];

        p.state[0] = s*cos(t);
        p.state[1] = s*sin(t);
        p.state[2] = z;
        p.state[3] = vpar_val;
        p.v_total = vtotal;
        p.v_perp = sqrt(vtotal*vtotal -  vpar_val*vpar_val);


        // printf("state:  %.15e, %.15e, %.15e, %.15e\n", p.state[0], p.state[1], p.state[2], p.state[3]);

        setup_particle(p, srange, trange, zrange, quad_pts, 1e-2, m, q, psi0);
        // printf("state:  %.15e, %.15e, %.15e, %.15e\n", p.state[0], p.state[1], p.state[2], p.state[3]);

        // printf("s=%.15e, t=%.15e, z=%.15e\n", s, t, z);
        // printf("y1=%.15e, y2=%.15e, z=%.15e\n", p.state[0], p.state[1], p.state[2]);
        // printf("xtemp=%.15e, %.15e, %.15e\n", p.x_temp[0], p.x_temp[1], p.x_temp[2]);

        calc_derivs(p, p.derivs, srange, trange, zrange, quad_pts, m, q, p.mu, psi0);

        // printf("derivs:  %.15e, %.15e, %.15e, %.15e\n", p.derivs[0], p.derivs[1], p.derivs[2], p.derivs[3]);


        out_arr[0] = p.derivs[0];
        out_arr[1] = p.derivs[1];
        out_arr[2] = p.derivs[2];
        out_arr[3] = p.derivs[3];


        // printf("modB=%.15e, modB_derivs=%.15e, %.15e, %.15e, G=%.15e, iota=%.15e\n", out_arr[0], out_arr[1], out_arr[2], out_arr[3], out_arr[4], out_arr[5]);


    }
}

extern "C" py::array_t<double> test_derivatives(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, double psi0, int n_points){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);

    py::buffer_info vpar_buf = vpar.request();
    double* vpar_arr = static_cast<double*>(vpar_buf.ptr);
    

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

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* vpar_d;
    cudaMalloc((void**)&vpar_d, vpar.size() * sizeof(double));
    cudaMemcpy(vpar_d, vpar_arr, vpar.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    cudaMalloc((void**)&out_d, 4*n_points * sizeof(double));



    int nthreads = 256;
    int nblks = n_points / nthreads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_derivs_kernel<<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, vpar_d, v_total, out_d, m, q, psi0, n_points);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "interpolation kernel time (ms): " << milliseconds<< "\n";
    
    double out[4*n_points];
    cudaMemcpy(&out, out_d, 4*n_points * sizeof(double), cudaMemcpyDeviceToHost);
    auto result = py::array_t<double>(4*n_points, out);
    return result;
}
