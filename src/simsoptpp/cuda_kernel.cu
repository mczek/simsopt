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

#include "magneticfield.h"
#include "boozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"

#define PARTICLES_PER_BLOCK 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


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
    double x_temp[4], x_err[4];
    double r_shape[4], phi_shape[4], z_shape[4];
    int i, j, k;
    double interpolation_loc[3];
    bool symmetry_exploited;
    int id;
    int step_attempt, step_accept;
    double surf_dist;
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




__host__ __device__ void shape(double& x, double& output, int i) {
    switch (i) {
        case 0:
            output = (1.0 - x) * (2.0 - x) * (3.0 - x) / 6.0;
            break;
        case 1:
            output = x * (2.0 - x) * (3.0 - x) / 2.0;
            break;
        case 2:
            output = x * (x - 1.0) * (3.0 - x) / 2.0;
            break;
        case 3:
            output = x * (x - 1.0) * (x - 2.0) / 6.0;
            break;
        default:
            output = 0.0;
            break;
    }
}


__host__ __device__ void shape(double x, double* shape){

    shape[0] = (1.0-x)*(2.0-x)*(3.0-x)/6.0;
    shape[1] = x*(2.0-x)*(3.0-x)/2.0;
    shape[2] = x*(x-1.0)*(3.0-x)/2.0;
    shape[3] = x*(x-1.0)*(x-2.0)/6.0;
    return;         
}

__device__ void interpolate(double*  out, const double* __restrict__ data, const int* index_i, const int* __restrict__ index_j, const int* __restrict__ index_k, 
    const double* __restrict__ r_shape, const double* __restrict__ phi_shape, const double* __restrict__ z_shape, int nphi, int nz, int n){

    // printf("particle %d: accessing index arrays at index: %d\n", threadIdx.x, threadIdx.x);
    int i = index_i[threadIdx.x];
    int j = index_j[threadIdx.x];
    int k = index_k[threadIdx.x];
    // printf("particle %d: i, j, k = %d, %d, %d\n", i, j, k);

    int cell_start = 64*(i*nphi*nz + j*nz + k);
    for(int ii=0; ii<=3; ++ii){ // s grid
        for(int jj=0; jj<=3; ++jj){ // theta grid           
            for(int kk=0; kk<=3; ++kk){ // zeta grid
                int row_idx = cell_start + 16*ii + 4*jj + kk;
                // printf("particle %d: row_index=%d\n", threadIdx.x, row_idx);               
                // printf("ii=%d, jj=%d, kk=%d, n=%d\n", ii, jj, kk, n);
                // printf("particle %d: accessing shape arrays at indices %d %d %d\n", 4*threadIdx.x + ii, 4*threadIdx.x + jj, 4*threadIdx.x+kk);
                double shape_val = r_shape[ii*PARTICLES_PER_BLOCK +threadIdx.x]*phi_shape[jj*PARTICLES_PER_BLOCK +threadIdx.x]*z_shape[kk*PARTICLES_PER_BLOCK +threadIdx.x];
                // printf("shape_val = %.15e\n", shape_val);
                for(int zz=0; zz<n; ++zz){
                    // printf("index access: %d\n", n*row_idx + zz);
                    // printf("particle %d: accessing out at index %d\n", threadIdx.x, PARTICLES_PER_BLOCK*zz + threadIdx.x);
                    // printf("particle %d: accessing data at index %d\n", threadIdx.x, n*row_idx +zz);
                    // if(threadIdx.x==1){
                    //     printf("particle %d: accessing data at index %d, value = %.15e, shape_val=%.15e\n", threadIdx.x, n*row_idx + zz, data[n*row_idx + zz], shape_val);
                    // }
                    out[PARTICLES_PER_BLOCK*zz + threadIdx.x] += data[n*row_idx + zz]*shape_val;
                    // printf("wrote to interpolant element %d\n", zz);
                }
            }
        }

    }
}

__device__ void calc_derivs(double* derivs, int deriv_id, double* quadpts_arr, double* x_temp, bool* symmetry_exploited, int* index_i, int* index_j, int* index_k, double* r_shape, double* phi_shape, double* z_shape,
                            double* mu, double m, double q, int nphi, int nz){

    __shared__ double block_interpolants[7*PARTICLES_PER_BLOCK];
    for(int i=0; i<7; ++i){
        block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
    }
    __syncthreads();
    interpolate(block_interpolants, quadpts_arr, index_i, index_j, index_k, r_shape, phi_shape, z_shape, nphi, nz, 7);
    __syncthreads();

    double x = x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double y = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    double z = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    double v_par = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];

    double B_r = block_interpolants[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double B_phi = block_interpolants[1*PARTICLES_PER_BLOCK + threadIdx.x];
    double B_z = block_interpolants[2*PARTICLES_PER_BLOCK + threadIdx.x];
    double GradAbsB_r = block_interpolants[3*PARTICLES_PER_BLOCK + threadIdx.x];
    double GradAbsB_phi = block_interpolants[4*PARTICLES_PER_BLOCK + threadIdx.x];
    double GradAbsB_z = block_interpolants[5*PARTICLES_PER_BLOCK + threadIdx.x];

    if(symmetry_exploited[threadIdx.x]){
        B_r *= -1.0;
        GradAbsB_phi *= -1.0;
        GradAbsB_z *= -1.0;
    }

    double phi = atan2(y, x);
    double B_x = cos(phi) * B_r - sin(phi) * B_phi;
    double B_y = sin(phi) * B_r + cos(phi) * B_phi;
    double GradAbsB_x = cos(phi) * GradAbsB_r - sin(phi) * GradAbsB_phi;
    double GradAbsB_y = sin(phi) * GradAbsB_r + cos(phi) * GradAbsB_phi;

    // printf("particle %d: B = %.15e, %.15e, %.15e, GradAbsB = %.15e, %.15e, %.15e\n", threadIdx.x, B_x, B_y, B_z, GradAbsB_x, GradAbsB_y, GradAbsB_z);

    double AbsB = sqrt(B_x*B_x + B_y*B_y + B_z*B_z);
    double v_perp2 = 2*mu[threadIdx.x]*AbsB;
    double fak1 = (v_par/AbsB);
    double fak2 = (m/(q*pow(AbsB, 3)))*(0.5*v_perp2 + v_par*v_par);

    // printf("particle %d: mu=%.15e, v_perp2 = %.15e, AbsB = %.15e, fak1 = %.15e, fak2 = %.15e\n", threadIdx.x, mu[threadIdx.x], v_perp2, AbsB, fak1, fak2);

    double BcrossGradAbsB_elt = B_y*GradAbsB_z - B_z*GradAbsB_y;
    derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_x + fak2*BcrossGradAbsB_elt;
    BcrossGradAbsB_elt = B_z*GradAbsB_x - B_x*GradAbsB_z;
    derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_y + fak2*BcrossGradAbsB_elt;
    BcrossGradAbsB_elt = B_x*GradAbsB_y - B_y*GradAbsB_x;
    derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x] = fak1*B_z + fak2*BcrossGradAbsB_elt;
    derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x] = -mu[threadIdx.x]*(B_x*GradAbsB_x + B_y*GradAbsB_y + B_z*GradAbsB_z)/AbsB;
    derivs[(6*deriv_id + 4)*PARTICLES_PER_BLOCK + threadIdx.x] = AbsB; // AbsB
    derivs[(6*deriv_id + 5)*PARTICLES_PER_BLOCK + threadIdx.x] = block_interpolants[6*PARTICLES_PER_BLOCK + threadIdx.x]; // boundary dist fn

    // printf("particle %d: derivative evaluated at %.15e, %.15e, %.15e, %.15e : %.15e, %.15e, %.15e, %.15e\n", 
        //    threadIdx.x, x, y, z, v_par, 
        //    derivs[(6*deriv_id + 0)*PARTICLES_PER_BLOCK + threadIdx.x], 
        //    derivs[(6*deriv_id + 1)*PARTICLES_PER_BLOCK + threadIdx.x], 
        //    derivs[(6*deriv_id + 2)*PARTICLES_PER_BLOCK + threadIdx.x], 
        //    derivs[(6*deriv_id + 3)*PARTICLES_PER_BLOCK + threadIdx.x]);


}
// out contains derivatives for x , y, z, v_par, and then norm of B and surface distance interpolation
__device__ void calc_derivs(particle_t& p, double* out, double* rrange_arr, double* phirange_arr, double* zrange_arr, double* quadpts_arr, double m, double q, double mu){
    /*
    * Returns     
    out[0] = ds/dtime
    out[1] = dtheta/dtime
    out[2] = dzeta/dtime

    out[3] = dvpar/dtime;
    out[4] = modB;
    

    */
    __shared__ int index_i[PARTICLES_PER_BLOCK];
    __shared__ int index_j[PARTICLES_PER_BLOCK];
    __shared__ int index_k[PARTICLES_PER_BLOCK];

    index_i[threadIdx.x] = p.i;
    index_j[threadIdx.x] = p.j;
    index_k[threadIdx.x] = p.k;

    __shared__ double r_shape[4*PARTICLES_PER_BLOCK];
    __shared__ double phi_shape[4*PARTICLES_PER_BLOCK];
    __shared__ double z_shape[4*PARTICLES_PER_BLOCK];

    for(int i=0; i<4; ++i){
        r_shape[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.r_shape[i];
        phi_shape[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.phi_shape[i];
        z_shape[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.z_shape[i];
    }

    __shared__ double block_interpolants[7*PARTICLES_PER_BLOCK];
    for(int i=0; i<7; ++i){
        block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
    }
    __syncthreads();
    int nphi = (phirange_arr[2]-1)/3;
    int nz = (zrange_arr[2]-1)/3;
    interpolate(block_interpolants, quadpts_arr, index_i, index_j, index_k, r_shape, phi_shape, z_shape, nphi, nz, 7);

    // double* loc = loc_shared + 3* block_part_id;
    // double interpolants[7] = {0.0};

    // interpolate(p, quadpts_arr, interpolants, rrange_arr, phirange_arr, zrange_arr, 7);

	//printf("interpolants:  %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", interpolants[0], interpolants[1], interpolants[2], interpolants[3], interpolants[4], interpolants[5]);
    

    double interpolants[7];
    for(int i=0; i<7; ++i){
        interpolants[i] = block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x];
    }

    double x = p.x_temp[0];
    double y = p.x_temp[1];
    double z = p.x_temp[2];
    double v_par = p.x_temp[3];
    if(p.symmetry_exploited){
        interpolants[0] *= -1.0;
        interpolants[4] *= -1.0;
        interpolants[5] *= -1.0;

    }


    double phi = atan2(y,x);

    // B_x = cos(phi) B_r - sin(phi) B_phi
    // B_y = sin(phi) B_r + cos(phi) B_phi
    double B_r = interpolants[0];
    double B_phi = interpolants[1];
    interpolants[0] = cos(phi) * B_r - sin(phi) * B_phi;
    interpolants[1] = sin(phi) * B_r + cos(phi) * B_phi;

    // GradAbsB_x = cos(phi) GradAbsB_r - sin(phi) GradAbsB_phi
    // GradAbsB_y = sin(phi) GradAbsB_r + cos(phi) GradAbsB_phi
    double GradAbsB_r = interpolants[3];
    double GradAbsB_phi = interpolants[4];
    interpolants[3] = cos(phi) * GradAbsB_r - sin(phi) * GradAbsB_phi;
    interpolants[4] = sin(phi) * GradAbsB_r + cos(phi) * GradAbsB_phi;

	//printf("B:  %.15e, %.15e, %.15e, GradAbsB: %.15e, %.15e, %.15e\n", interpolants[0], interpolants[1], interpolants[2], interpolants[3], interpolants[4], interpolants[5]);
    // interpolants now stores B, GradAbsB, signed dist fn
    // abs B = || B ||
    double AbsB = interpolants[0]*interpolants[0] + interpolants[1]*interpolants[1] + interpolants[2]*interpolants[2];
    AbsB = sqrt(AbsB);

    double v_perp2 = 2*p.mu*AbsB;
    // p.v_perp = sqrt(v_perp2);
    double fak1 = (v_par/AbsB);
    double fak2 = (m/(q*pow(AbsB, 3)))*(0.5*v_perp2 + v_par*v_par);

    double* B = interpolants;
    double* GradAbsB = interpolants + 3;

    double BcrossGradAbsB_elt = B[1]*GradAbsB[2] - B[2]*GradAbsB[1];
    out[0] = fak1*B[0] + fak2*BcrossGradAbsB_elt;
    
    BcrossGradAbsB_elt = B[2]*GradAbsB[0] - B[0]*GradAbsB[2];
    out[1] = fak1*B[1] + fak2*BcrossGradAbsB_elt;

    BcrossGradAbsB_elt = B[0]*GradAbsB[1] - B[1]*GradAbsB[0];
    out[2] = fak1*B[2] + fak2*BcrossGradAbsB_elt;

    out[3] = -p.mu*(B[0]*GradAbsB[0] + B[1]*GradAbsB[1] + B[2]*GradAbsB[2])/AbsB;      

    out[4] = AbsB; // AbsB
    out[5] = interpolants[6]; // boundary dist fn

    p.surf_dist = interpolants[6];

	//printf("derivative evaluated at %.15e, %.15e, %.15e, %.15e : %.15e, %.15e, %.15e, %.15e\n", x, y, z, v_par, out[0], out[1], out[2], out[3]);
	//printf("interpolant values : %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", interpolants[0], interpolants[1], interpolants[2], interpolants[3], interpolants[4], interpolants[5], interpolants[6]);

}

__device__ void build_state(double* x_temp, int deriv_id, bool* symmetry_exploited, int* index_i, int* index_j, int* index_k,
                            double* r_shape, double* phi_shape, double* z_shape, double* state, double* derivs, double* dt,
                            double* rrange_arr, double* phirange_arr, double* zrange_arr){
    const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    double wgts[6] = {0.0}; 
    for (int i = 0; i < 4; i++) {
        x_temp[i*PARTICLES_PER_BLOCK + threadIdx.x] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
    }
    
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

    for (int j=0; j<6; ++j){
        for(int i=0; i<4; ++i){
            // printf("contribution: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", p.dt, wgts[j], p.derivs[6*j+i], p.dt * wgts[j], wgts[j] * p.derivs[6*j+i], p.dt * p.derivs[6*j+i], p.dt * wgts[j] * p.derivs[6*j+i]);
            x_temp[i*PARTICLES_PER_BLOCK + threadIdx.x] += dt[threadIdx.x] * wgts[j] * derivs[(6*j+i)*PARTICLES_PER_BLOCK + threadIdx.x];
            // printf("contribution: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", dt[threadIdx.x], wgts[j], derivs[(6*j+i)*PARTICLES_PER_BLOCK + threadIdx.x], dt[threadIdx.x] * wgts[j], wgts[j] * derivs[(6*j+i)*PARTICLES_PER_BLOCK + threadIdx.x], dt[threadIdx.x] * derivs[(6*j+i)*PARTICLES_PER_BLOCK + threadIdx.x], dt[threadIdx.x] * wgts[j] * derivs[(6*j+i)*PARTICLES_PER_BLOCK + threadIdx.x]);
        }
    } 

    double x = x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double y = x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x];
    double z = x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x];
    double v_par = x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x];


    // convert to cylindrical coordinates for interpolation
    double r = sqrt(x*x + y*y);
    double phi = atan2(y, x);
    

    // printf("deriv pt: %.15e, %.15e, %.15e, %.15e, %.15e\n", p.dt, r, phi, z, p.x_temp[3]);

    // restrict phi to [0, 2pi / nfp]
    double period = phirange_arr[1];
    phi = fmod(phi, period);
    phi += period*(phi < 0);

    // exploit stellarator symmetry
    symmetry_exploited[threadIdx.x] = z < 0;
    if(symmetry_exploited[threadIdx.x]){
        z = -z;
        phi = 2*M_PI - phi;
        phi = fmod(phi, period);
        phi += period*(phi < 0);
    }

    // note: remove this memory usage
    // p.interpolation_loc[0] = r;
    // p.interpolation_loc[1] = phi;
    // p.interpolation_loc[2] = z;

    // printf("deriv pt: r=%.15e, phi=%.15e, z=%.15e\n", r, phi, z);

    /*
    * index into the grid and calculate weights
    */ 
    double r_grid_size = (rrange_arr[1]-rrange_arr[0]) / (rrange_arr[2]-1);
    double phi_grid_size = (phirange_arr[1]-phirange_arr[0]) / (phirange_arr[2]-1);
    double z_grid_size = (zrange_arr[1]-zrange_arr[0]) / (zrange_arr[2]-1);

    // printf("grid sizes: r=%.15e, phi=%.15e, zeta=%.15e\n", r_grid_size, phi_grid_size, z_grid_size);

    int i = 3*((int) ((r - rrange_arr[0]) / r_grid_size) / 3);
    int j = 3*((int) ((phi - phirange_arr[0]) / phi_grid_size) / 3);
    int k = 3*((int) ((z - zrange_arr[0]) / z_grid_size) / 3);

    i = min(i, (int)rrange_arr[2]-4);
    j = min(j, (int)phirange_arr[2]-4);
    k = min(k, (int)zrange_arr[2]-4);

    i = max(i, 0); // if r too small to be in the device, extrapolate



    // printf("x=%.15e, y=%.15e, z=%.15e, deriv pt: r=%.15e, phi=%.15e, z=%.15e interpolant indices: i=%d, j=%d, k=%d\n", p.x_temp[0], p.x_temp[1], p.x_temp[2], r, phi, z, p.i, p.j, p.k);

    // normalized positions in local grid wrt e.g. r at index i
    // maps the position to [0,3] in the "meta grid"
    double r_rel = (r -  i*r_grid_size - rrange_arr[0]) / r_grid_size;
    double phi_rel = (phi -  j*phi_grid_size - phirange_arr[0]) / phi_grid_size;
    double z_rel = (z - k*z_grid_size - zrange_arr[0]) / z_grid_size;
   
    for(int i=0; i<4; ++i){
        shape(r_rel, r_shape[i*PARTICLES_PER_BLOCK + threadIdx.x], i);
        shape(phi_rel, phi_shape[i*PARTICLES_PER_BLOCK + threadIdx.x], i);
        shape(z_rel, z_shape[i*PARTICLES_PER_BLOCK + threadIdx.x], i);
    }

    // convert to cell id
    index_i[threadIdx.x] = i/3;
    index_j[threadIdx.x] = j/3;
    index_k[threadIdx.x] = k/3;
    


}

__device__ void build_state(particle_t& p, int deriv_id, double* rrange_arr, double* phirange_arr, double* zrange_arr){
   

    const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    double wgts[6] = {0.0}; 

    for (int i = 0; i < 4; i++) {
        p.x_temp[i] = p.state[i];
    }

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

    for (int j=0; j<6; ++j){
        for(int i=0; i<4; ++i){
            // printf("contribution: %.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e\n", p.dt, wgts[j], p.derivs[6*j+i], p.dt * wgts[j], wgts[j] * p.derivs[6*j+i], p.dt * p.derivs[6*j+i], p.dt * wgts[j] * p.derivs[6*j+i]);
            p.x_temp[i] += p.dt * wgts[j] * p.derivs[6*j+i];
        }
    } 

    // convert to cylindrical coordinates for interpolation
    double r = sqrt(p.x_temp[0]*p.x_temp[0] + p.x_temp[1]*p.x_temp[1]);
    double phi = atan2(p.x_temp[1], p.x_temp[0]);
    double z = p.x_temp[2];
    double v_par = p.x_temp[3];
    

    // printf("deriv pt: %.15e, %.15e, %.15e, %.15e, %.15e\n", p.dt, r, phi, z, p.x_temp[3]);

    // restrict phi to [0, 2pi / nfp]
    double period = phirange_arr[1];
    phi = fmod(phi, period);
    phi += period*(phi < 0);

    // exploit stellarator symmetry
    p.symmetry_exploited = z < 0;
    if(p.symmetry_exploited){
        z = -z;
        phi = 2*M_PI - phi;
        phi = fmod(phi, period);
        phi += period*(phi < 0);
    }

    // note: remove this memory usage
    // p.interpolation_loc[0] = r;
    // p.interpolation_loc[1] = phi;
    // p.interpolation_loc[2] = z;

    // printf("deriv pt: r=%.15e, phi=%.15e, z=%.15e\n", r, phi, z);

    /*
    * index into the grid and calculate weights
    */ 
    double r_grid_size = (rrange_arr[1]-rrange_arr[0]) / (rrange_arr[2]-1);
    double phi_grid_size = (phirange_arr[1]-phirange_arr[0]) / (phirange_arr[2]-1);
    double z_grid_size = (zrange_arr[1]-zrange_arr[0]) / (zrange_arr[2]-1);

    // printf("grid sizes: r=%.15e, phi=%.15e, zeta=%.15e\n", r_grid_size, phi_grid_size, z_grid_size);

    p.i = 3*((int) ((r - rrange_arr[0]) / r_grid_size) / 3);
    p.j = 3*((int) ((phi - phirange_arr[0]) / phi_grid_size) / 3);
    p.k = 3*((int) ((z - zrange_arr[0]) / z_grid_size) / 3);

    p.i = min(p.i, (int)rrange_arr[2]-4);
    p.j = min(p.j, (int)phirange_arr[2]-4);
    p.k = min(p.k, (int)zrange_arr[2]-4);

    p.i = max(p.i, 0); // if r too small to be in the device, extrapolate



    // printf("x=%.15e, y=%.15e, z=%.15e, deriv pt: r=%.15e, phi=%.15e, z=%.15e interpolant indices: i=%d, j=%d, k=%d\n", p.x_temp[0], p.x_temp[1], p.x_temp[2], r, phi, z, p.i, p.j, p.k);

    // normalized positions in local grid wrt e.g. r at index i
    // maps the position to [0,3] in the "meta grid"
    double r_rel = (r -  p.i*r_grid_size - rrange_arr[0]) / r_grid_size;
    double phi_rel = (phi -  p.j*phi_grid_size - phirange_arr[0]) / phi_grid_size;
    double z_rel = (z - p.k*z_grid_size - zrange_arr[0]) / z_grid_size;
   
    shape(r_rel, p.r_shape);
    shape(phi_rel, p.phi_shape);
    shape(z_rel, p.z_shape);

    // convert to cell id
    p.i /= 3;
    p.j /= 3;
    p.k /= 3;

}

__device__ void setup_particle(double* mu, double* t, double* dt, double* dtmax, double* x_temp, bool* symmetry_exploited, int* index_i, int* index_j, int* index_k,
                            double* quad_pts, double* r_shape, double* phi_shape, double* z_shape, double* state, double* derivs,
                            double* rrange_arr, double* phirange_arr, double* zrange_arr, double vtotal, double tmax, double m, double q){
    t[threadIdx.x] = 0.0;
    dt[threadIdx.x] = 0.0;
    symmetry_exploited[threadIdx.x] = false;
    build_state(x_temp, 0, symmetry_exploited, index_i, index_j, index_k,
                            r_shape, phi_shape, z_shape, state, derivs, dt,
                            rrange_arr, phirange_arr, zrange_arr);
    // dummy call to get norm B
    mu[threadIdx.x] = -1.0; // initialize mu
    int nphi = (phirange_arr[2]-1)/3;
    int nz = (zrange_arr[2]-1)/3;
    calc_derivs(derivs, 0, quad_pts, x_temp, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, mu, m, q, nphi, nz);

    double v_par = state[3*PARTICLES_PER_BLOCK + threadIdx.x];
    double v_perp2 = vtotal*vtotal - v_par*v_par;
    double modB = derivs[4*PARTICLES_PER_BLOCK + threadIdx.x];
    double denom = 1 / (2*modB);
    mu[threadIdx.x] = v_perp2 * denom;

    double x = state[0*PARTICLES_PER_BLOCK + threadIdx.x];
    double y = state[1*PARTICLES_PER_BLOCK + threadIdx.x];
    // can at most do quarter of a revolution per step
    double r = sqrt(x*x + y*y);
    dtmax[threadIdx.x] = r*0.5*M_PI/vtotal;
    dt[threadIdx.x] = 1e-3*dtmax[threadIdx.x];
    // printf("initial dt = %.15e, r = %.15e, v_total = %.15e\n", dt[threadIdx.x], r, vtotal);

}

// set initial time step, calculate mu
__device__ void setup_particle(particle_t& p, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                         double tmax, double m, double q){
                             // double mu;
    p.t = 0.0;
    p.dt = 0.0;
    build_state(p, 0, srange_arr, trange_arr, zrange_arr);

    // dummy call to get norm B
    calc_derivs(p, p.derivs, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, -1);

    double v_perp2 = p.v_perp*p.v_perp;
    double denom = 1 / (2*p.derivs[4]);
    p.mu = v_perp2 * denom;
    // printf("in setup_particle mu = %.15e, v_perp2=%.15e, denom=%.15e, AbsB=%.15e\n", p.mu, v_perp2, denom, p.derivs[4]);

    // can at most do quarter of a revolution per step
    double r = sqrt(p.state[0]*p.state[0] + p.state[1]*p.state[1]);
    double v_total = sqrt(v_perp2 + p.state[3]*p.state[3]);
    p.dtmax = r*0.5*M_PI/v_total;
    // p.dtmax = 0.5*M_PI*abs(p.derivs[5]) / (p.derivs[4]*p.v_total);
    p.dt = 1e-3*p.dtmax;
	//printf("initial dt = %.15e, r = %.15e, v_total = %.15e\n", p.dt, r, v_total);
}

__device__ void adjust_time(double* t, double* dt, double* state, double* derivs, double* x_temp, bool* has_left, double atol, double rtol, double tmax, double* dtmax){
    if(has_left[threadIdx.x]){
        return;
    }
    const double bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;
    // Compute  error
    // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // resolve typo in boost docs: https://numerical.recipes/book.html
    double max_err = 0.0;
    double err_elt;
    for(int i = 0; i < 4; i++) {
        double state_i = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        double deriv_i = derivs[(6*0 + i)*PARTICLES_PER_BLOCK + threadIdx.x];
        err_elt = dt[threadIdx.x]*(bhat1 * deriv_i
                                 + bhat3 * derivs[(6*2 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat4 * derivs[(6*3 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat5 * derivs[(6*4 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat6 * derivs[(6*5 + i)*PARTICLES_PER_BLOCK + threadIdx.x] 
                                 + bhat7 * derivs[(6*6 + i)*PARTICLES_PER_BLOCK + threadIdx.x]);
        err_elt = fabs(err_elt) / (atol + rtol*(fabs(state_i) + dt[threadIdx.x]*fabs(deriv_i)));
        max_err = fmax(max_err, err_elt);
    }

    // Compute new step size
    double dt_new = dt[threadIdx.x]*0.9*pow(max_err, -1.0/3.0);
    dt_new = fmax(dt_new, 0.2 * dt[threadIdx.x]);  // Limit step size reduction
    dt_new = fmin(dt_new, 5.0 * dt[threadIdx.x]);  // Limit step size increase
    dt_new = fmin(dtmax[threadIdx.x], dt_new);
    if ((0.5 < max_err) & (max_err < 1.0)){
        dt_new = dt[threadIdx.x];
    }

    if(max_err <= 1.0) {
        // Accept the step
        t[threadIdx.x] += dt[threadIdx.x];
        dt[threadIdx.x] = fmin(dt_new, tmax - t[threadIdx.x]);

        for(int i = 0; i < 4; i++) {
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = x_temp[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }

        
        has_left[threadIdx.x] = derivs[(6*6 + 5)*PARTICLES_PER_BLOCK + threadIdx.x] < 0; // boundary dist fn at new location
    } else {
        // Reject the step and try again with smaller dt
        dt[threadIdx.x] = dt_new;
    }
}

__device__ void adjust_time(particle_t& p, double tmax){
    if(p.has_left){
        return;
    }

    const double bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;

    // Compute  error
    // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // resolve typo in boost docs: https://numerical.recipes/book.html
    double atol=1e-9;
    double rtol=1e-9;
    double err = 0.0;
    bool accept = true;
    for (int i = 0; i < 4; i++) {
        p.x_err[i] = p.dt*(bhat1 * p.derivs[i] + bhat3 * p.derivs[12+i] + bhat4 * p.derivs[18+i] + bhat5 * p.derivs[24+i] + bhat6 * p.derivs[30+i] + bhat7 * p.derivs[36+i]);
       
        // if(i==3){
        //     atol *= 1e5;
        // }
        p.x_err[i] = fabs(p.x_err[i]) / (atol + rtol*(fabs(p.state[i]) + p.dt*fabs(p.derivs[i])));      
        err = fmax(err, p.x_err[i]);
    }

    // Compute new step size
    double dt_new = p.dt*0.9*pow(err, -1.0/3.0);
    dt_new = fmax(dt_new, 0.2 * p.dt);  // Limit step size reduction
    dt_new = fmin(dt_new, 5.0 * p.dt);  // Limit step size increase
    dt_new = fmin(p.dtmax, dt_new);
    if ((0.5 < err) & (err < 1.0)){
        dt_new = p.dt;
    }
    p.step_attempt++;

	//printf("err = %.15e\n", err);

    if (err <= 1.0) {
        // Accept the step
        p.t += p.dt;
        p.dt = fmin(dt_new, tmax - p.t);

        p.state[0] = p.x_temp[0];
        p.state[1] = p.x_temp[1];
        p.state[2] = p.x_temp[2];
        p.state[3] = p.x_temp[3];

        p.has_left = p.surf_dist < 0;
        p.step_accept++;


    } else {
        // Reject the step and try again with smaller dt
        p.dt = dt_new;
    }

}

__device__ void trace_particle(double* state, double* derivs, double* dt, double* t, double* dtmax, double vtotal, double* x_temp, bool* has_left, double* mu, double tmax, 
                    bool* symmetry_exploited, int* index_i, int* index_j, int* index_k,
                    double* quadpts_arr, double* r_shape, double* phi_shape, double* z_shape,
                    double* srange_arr, double* trange_arr, double* zrange_arr, double m, double q){
    setup_particle(mu, t, dt, dtmax, x_temp, symmetry_exploited, index_i, index_j, index_k,
                    quadpts_arr, r_shape, phi_shape, z_shape, state, derivs,
                    srange_arr, trange_arr, zrange_arr, vtotal, tmax, m, q);
    int nphi = (trange_arr[2]-1)/3;
    int nz = (zrange_arr[2]-1)/3;
    while(t[threadIdx.x] < tmax){
        // printf("particle %d at time %.15e\n", threadIdx.x, t[threadIdx.x]);
        for(int k=0; k<7; ++k){
            build_state(x_temp, k, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, state, derivs, dt,
                        srange_arr, trange_arr, zrange_arr);

            calc_derivs(derivs, k, quadpts_arr, x_temp, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, mu, m, q, 
                        nphi, nz);
        }
        double atol=1e-9;
        double rtol=1e-9;
        adjust_time(t, dt, state, derivs, x_temp, has_left, atol, rtol, 1e-2, dtmax);
        if(has_left[threadIdx.x]){
            // printf("particle %d has left the device at time %.15e\n", threadIdx.x, t[threadIdx.x]);
            return;
        }
    }
}


__device__    void trace_particle(particle_t& p, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                         double tmax, double m, double q){


    setup_particle(p, srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q);
    int counter = 0;
    while(p.t < tmax){
        for(int k=0; k<7; ++k){
            build_state(p, k, srange_arr, trange_arr, zrange_arr);
            calc_derivs(p, p.derivs + 6*k, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, p.mu);
        }
        adjust_time(p, tmax);
        
        if(p.has_left){
            return;
        }
        counter++;
    }
    return;
}

__global__ void particle_trace_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double tmax, double m, double q, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        particle_t p = particles[idx];

        // printf("v_perp = %.15e, v_par = %.15e, v_total = %.15e\n", p.v_perp, p.state[3], p.v_total);

        __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
        __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
        __shared__ double dt[PARTICLES_PER_BLOCK];
        __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
        __shared__ int index_i[PARTICLES_PER_BLOCK];
        __shared__ int index_j[PARTICLES_PER_BLOCK];
        __shared__ int index_k[PARTICLES_PER_BLOCK];
        __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double mu[PARTICLES_PER_BLOCK];
        __shared__ double t[PARTICLES_PER_BLOCK];
        __shared__ double dtmax[PARTICLES_PER_BLOCK];
        __shared__ double state[4 * PARTICLES_PER_BLOCK];
        __shared__ bool has_left[PARTICLES_PER_BLOCK];

        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }
        trace_particle(state, derivs, dt, t, dtmax, p.v_total, x_temp, has_left, mu, tmax,
                        symmetry_exploited, index_i, index_j, index_k,
                        quadpts_arr, r_shape, phi_shape, z_shape,
                        srange_arr, trange_arr, zrange_arr, m, q);
        // trace_particle(particles[idx], srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q);

        particles[idx].dt = dt[threadIdx.x];
        particles[idx].t = t[threadIdx.x];
        particles[idx].has_left = has_left[threadIdx.x];
        for(int i=0; i<4; ++i){
            particles[idx].state[i] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
    }
}


extern "C" vector<double> gpu_tracing(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, int nparticles){

    //  read data in from python
    py::buffer_info stz_init_buf = stz_init.request();
    double* stz_init_arr = static_cast<double*>(stz_init_buf.ptr);
    // auto ptr = stz_init.data();
    // int size = stz_init.size();
    // double stz_init_arr[size];
    // std::memcpy(stz_init_arr, ptr, size * sizeof(double));
    
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

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double s = stz_init_arr[start];
        double theta = stz_init_arr[start+1];
        
        // convert to alternative coordinates
        particles[i].state[0] = stz_init_arr[start]; // x
        particles[i].state[1] = stz_init_arr[start+1];// y
        
        particles[i].state[2] = stz_init_arr[start+2]; // z
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;

    }

   
    particle_t* particles_d;
    gpuErrchk(cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t)) );
    gpuErrchk(cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice) );

    double* srange_d;
    gpuErrchk(cudaMalloc((void**)&srange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* zrange_d;
    gpuErrchk(cudaMalloc((void**)&zrange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);


    double* quadpts_d;
    gpuErrchk(cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double)) ); 
    gpuErrchk(cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice) );

    int nthreads = PARTICLES_PER_BLOCK;
    int nblks = nparticles / nthreads + 1;
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    particle_trace_kernel<<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, tmax, m, q, nparticles);

    gpuErrchk(cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost) );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";

    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        // last location in Boozer coordinates
        particle_output[7*i] = y1;
        particle_output[7*i + 1] = y2;
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
        double x = loc_arr[0]*cos(loc_arr[1]);
        double y = loc_arr[0]*sin(loc_arr[1]);
        double z = loc_arr[2];




        p.state[0] = x;
        p.state[1] = y;
        p.state[2] = z;

        p.dt = 1e-3; //needed for build_state

        // printf("x=%.15e, y=%.15e, z=%.15e\n", p.state[0], p.state[1], p.state[2]);
        __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
        __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
        __shared__ int index_i[PARTICLES_PER_BLOCK];
        __shared__ int index_j[PARTICLES_PER_BLOCK];
        __shared__ int index_k[PARTICLES_PER_BLOCK];
        __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double state[4 * PARTICLES_PER_BLOCK];
        __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
        __shared__ double dt[PARTICLES_PER_BLOCK];
        dt[threadIdx.x] = 1e-3; // needed for build_state
        symmetry_exploited[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }
        build_state(x_temp, 0, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, state, derivs, dt, srange, trange, zrange);
        __syncthreads();

        // if(threadIdx.x == 1){
        //     // printf("build_state complete on particle %d\n", idx);
        //     printf("x_temp: %.15e, %.15e, %.15e, %.15e\n", x_temp[0*PARTICLES_PER_BLOCK + threadIdx.x], x_temp[1*PARTICLES_PER_BLOCK + threadIdx.x], x_temp[2*PARTICLES_PER_BLOCK + threadIdx.x], x_temp[3*PARTICLES_PER_BLOCK + threadIdx.x]);
        //     printf("symmetry_exploited: %d\n", symmetry_exploited[threadIdx.x]);
        //     printf("index_i: %d, index_j: %d, index_k: %d\n", index_i[threadIdx.x], index_j[threadIdx.x], index_k[threadIdx.x]);
        //     printf("r_shape: %.15e, %.15e, %.15e, %.15e\n", r_shape[0*PARTICLES_PER_BLOCK + threadIdx.x], r_shape[1*PARTICLES_PER_BLOCK + threadIdx.x], r_shape[2*PARTICLES_PER_BLOCK + threadIdx.x], r_shape[3*PARTICLES_PER_BLOCK + threadIdx.x]);
        //     printf("phi_shape: %.15e, %.15e, %.15e, %.15e\n", phi_shape[0*PARTICLES_PER_BLOCK + threadIdx.x], phi_shape[1*PARTICLES_PER_BLOCK + threadIdx.x], phi_shape[2*PARTICLES_PER_BLOCK + threadIdx.x], phi_shape[3*PARTICLES_PER_BLOCK + threadIdx.x]);
        //     printf("z_shape: %.15e, %.15e, %.15e, %.15e\n", z_shape[0*PARTICLES_PER_BLOCK + threadIdx.x], z_shape[1*PARTICLES_PER_BLOCK + threadIdx.x], z_shape[2*PARTICLES_PER_BLOCK + threadIdx.x], z_shape[3*PARTICLES_PER_BLOCK + threadIdx.x]);


        // }

        // printf("build_state complete on particle %d\n", idx);

        // __shared__ int index_i[PARTICLES_PER_BLOCK];
        // __shared__ int index_j[PARTICLES_PER_BLOCK];
        // __shared__ int index_k[PARTICLES_PER_BLOCK];

        // index_i[threadIdx.x] = p.i;
        // index_j[threadIdx.x] = p.j;
        // index_k[threadIdx.x] = p.k;

        // __shared__ double r_shape[4*PARTICLES_PER_BLOCK];
        // __shared__ double phi_shape[4*PARTICLES_PER_BLOCK];
        // __shared__ double z_shape[4*PARTICLES_PER_BLOCK];

        // for(int i=0; i<4; ++i){
        //     r_shape[4*threadIdx.x + i] = p.r_shape[i];
        //     phi_shape[4*threadIdx.x + i] = p.phi_shape[i];
        //     z_shape[4*threadIdx.x + i] = p.z_shape[i];
        // }

        __shared__ double block_interpolants[7*PARTICLES_PER_BLOCK];
        for(int i=0; i<7; ++i){
            block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x] = 0.0;
        }
        
        // printf("calling interpolate for particle %d\n", threadIdx.x);
        int nphi = (trange[2]-1)/3;
        int nz = (zrange[2]-1)/3;
        interpolate(block_interpolants, quad_pts, index_i, index_j, index_k, r_shape, phi_shape, z_shape, nphi, nz, 7);
        // printf("returned from interpolate for particle %d\n", threadIdx.x);
        // interpolate(p, quad_pts, out_arr, srange, trange, zrange, n);

        for(int i=0; i<7; ++i){
            out_arr[i] = block_interpolants[i*PARTICLES_PER_BLOCK + threadIdx.x];

            // if(threadIdx.x==1){
            //     printf("interpolated value %d for particle %d: %.15e\n", i, idx, out_arr[i]);
            // }
        }

        if(symmetry_exploited[threadIdx.x]){
            out_arr[0] *= -1.0;
            out_arr[4] *= -1.0;
            out_arr[5] *= -1.0;

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



    int nthreads = PARTICLES_PER_BLOCK;
    int nblks = n_points / nthreads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_interpolation_kernel<<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, out_d, n, n_points);
    
    double out[n*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, n*n_points * sizeof(double), cudaMemcpyDeviceToHost) );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "interpolation kernel time (ms): " << milliseconds<< "\n";
    

    auto result = py::array_t<double>(n*n_points, out);
    return result;

}


__global__ void test_gpu_derivs_kernel(double* quad_pts, double* srange, double* trange, double* zrange, double* loc, double* vpar, double vtotal, double* out, double m, double q, int n_points){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n_points){
        double* loc_arr = loc + 3*idx;
        double* out_arr  =  out + 4*idx;
        double vpar_val = vpar[idx];

        particle_t p;
        double r = loc_arr[0];
        double phi = loc_arr[1];
        double z = loc_arr[2];

        p.state[0] = r*cos(phi);
        p.state[1] = r*sin(phi);
        p.state[2] = z;
        p.state[3] = vpar_val;
        p.v_total = vtotal;
        p.v_perp = sqrt(vtotal*vtotal -  vpar_val*vpar_val);
        // printf("v_perp = %.15e, v_par = %.15e, v_total = %.15e\n", p.v_perp, p.state[3], p.v_total);

        __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
        __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
        __shared__ double dt[PARTICLES_PER_BLOCK];
        __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
        __shared__ int index_i[PARTICLES_PER_BLOCK];
        __shared__ int index_j[PARTICLES_PER_BLOCK];
        __shared__ int index_k[PARTICLES_PER_BLOCK];
        __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double mu[PARTICLES_PER_BLOCK];
        __shared__ double t[PARTICLES_PER_BLOCK];
        __shared__ double dtmax[PARTICLES_PER_BLOCK];
        __shared__ double state[4 * PARTICLES_PER_BLOCK];
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }

        setup_particle(mu, t, dt, dtmax, x_temp, symmetry_exploited, index_i, index_j, index_k,
                            quad_pts, r_shape, phi_shape, z_shape, state, derivs,
                            srange, trange, zrange, p.v_total, 1e-2, m, q);

        // setup_particle(p, srange, trange, zrange, quad_pts, 1e-2, m, q);



        // for(int i=0; i<4; ++i){
        //     x_temp[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.x_temp[i];
        // }
        // dt[threadIdx.x] = p.dt;
        // symmetry_exploited[threadIdx.x] = p.symmetry_exploited;
        // index_i[threadIdx.x] = p.i;
        // index_j[threadIdx.x] = p.j;
        // index_k[threadIdx.x] = p.k;
        // for(int i=0; i<4; ++i){
        //     r_shape[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.r_shape[i];
        //     phi_shape[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.phi_shape[i];
        //     z_shape[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.z_shape[i];
        // }
        // mu[threadIdx.x] = p.mu;

        // printf("mu value stored in shared memory: %.15e\n", mu[threadIdx.x]);

        int nphi = (trange[2]-1)/3;
        int nz = (zrange[2]-1)/3;
        calc_derivs(derivs, 0, quad_pts, x_temp, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, mu, m, q, nphi, nz);

        // copy back
        for(int i=0; i<4; ++i){
            p.derivs[i] = derivs[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
        // calc_derivs(p, p.derivs, srange, trange, zrange, quad_pts, m, q, p.mu);

        out_arr[0] = p.derivs[0];
        out_arr[1] = p.derivs[1];
        out_arr[2] = p.derivs[2];
        out_arr[3] = p.derivs[3];

    }
}

extern "C" py::array_t<double> test_derivatives(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, int n_points){
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



    int nthreads = PARTICLES_PER_BLOCK;
    int nblks = n_points / nthreads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_derivs_kernel<<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, vpar_d, v_total, out_d, m, q, n_points);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "interpolation kernel time (ms): " << milliseconds<< "\n";
    
    double out[4*n_points];
    gpuErrchk( cudaMemcpy(&out, out_d, 4*n_points * sizeof(double), cudaMemcpyDeviceToHost) );
    auto result = py::array_t<double>(4*n_points, out);
    return result;
}

__global__ void test_gpu_timestep_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double m, double q, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        particle_t p = particles[idx];

        // printf("v_perp = %.15e, v_par = %.15e, v_total = %.15e\n", p.v_perp, p.state[3], p.v_total);

        __shared__ double x_temp[4 * PARTICLES_PER_BLOCK];
        __shared__ double derivs[42 * PARTICLES_PER_BLOCK];
        __shared__ double dt[PARTICLES_PER_BLOCK];
        __shared__ bool symmetry_exploited[PARTICLES_PER_BLOCK];
        __shared__ int index_i[PARTICLES_PER_BLOCK];
        __shared__ int index_j[PARTICLES_PER_BLOCK];
        __shared__ int index_k[PARTICLES_PER_BLOCK];
        __shared__ double r_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double phi_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double z_shape[4 * PARTICLES_PER_BLOCK];
        __shared__ double mu[PARTICLES_PER_BLOCK];
        __shared__ double t[PARTICLES_PER_BLOCK];
        __shared__ double dtmax[PARTICLES_PER_BLOCK];
        __shared__ double state[4 * PARTICLES_PER_BLOCK];
        __shared__ bool has_left[PARTICLES_PER_BLOCK];

        has_left[threadIdx.x] = false;
        for(int i=0; i<4; ++i){
            state[i*PARTICLES_PER_BLOCK + threadIdx.x] = p.state[i];
        }

        setup_particle(mu, t, dt, dtmax, x_temp, symmetry_exploited, index_i, index_j, index_k,
                            quadpts_arr, r_shape, phi_shape, z_shape, state, derivs,
                            srange_arr, trange_arr, zrange_arr, p.v_total, 1e-2, m, q);
    // printf("tracing particle %d\n", idx);
        int nphi = (trange_arr[2]-1)/3;
        int nz = (zrange_arr[2]-1)/3;
        // setup_particle(particles[idx], srange_arr, trange_arr, zrange_arr, quadpts_arr, 1e-2, m, q);
        while(t[threadIdx.x] == 0.0){
            for(int k=0; k<7; ++k){
                // printf("building state %d\n", k);
                build_state(x_temp, k, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, state, derivs, dt,
                            srange_arr, trange_arr, zrange_arr);
                // build_state(particles[idx], k, srange_arr, trange_arr, zrange_arr);
                // printf("calclulating derivative %d\n", k);
                calc_derivs(derivs, k, quadpts_arr, x_temp, symmetry_exploited, index_i, index_j, index_k, r_shape, phi_shape, z_shape, mu, m, q, 
                            nphi, nz);
                // calc_derivs(particles[idx], particles[idx].derivs + 6*k, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, particles[idx].mu);
            }
            // adjust_time(particles[idx], 1e-2);
            double atol=1e-9;
            double rtol=1e-9;
            adjust_time(t, dt, state, derivs, x_temp, has_left, atol, rtol, 1e-2, dtmax);
        }
        // printf("tracing particle %d finished at t=%.15e\n", idx, particles[idx].t);
        particles[idx].dt = dt[threadIdx.x];
        particles[idx].t = t[threadIdx.x];
        particles[idx].has_left = has_left[threadIdx.x];
        for(int i=0; i<4; ++i){
            particles[idx].state[i] = state[i*PARTICLES_PER_BLOCK + threadIdx.x];
        }
    }
    return;
}



extern "C" vector<double> test_timestep(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, int nparticles){

    //  read data in from python
    py::buffer_info stz_init_buf = stz_init.request();
    double* stz_init_arr = static_cast<double*>(stz_init_buf.ptr);

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

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double r = stz_init_arr[start];
        double phi = stz_init_arr[start+1];
        
        // convert to cartesian coordinates
        particles[i].state[0] = r*cos(phi); // x
        particles[i].state[1] = r*sin(phi); // y
        particles[i].state[2] = stz_init_arr[start+2]; // z
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0.0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;
    }
    
    particle_t* particles_d;
    gpuErrchk( cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t)) );
    gpuErrchk( cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice) );

    double* srange_d;
    gpuErrchk( cudaMalloc((void**)&srange_d, 3 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* zrange_d;
    gpuErrchk( cudaMalloc((void**)&zrange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );

    double* trange_d;
    gpuErrchk(cudaMalloc((void**)&trange_d, 3 * sizeof(double)) );
    gpuErrchk(cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice) );


    double* quadpts_d;

    // std::cout << "quadpts.size() = " << quad_pts.size() << "\n";
    gpuErrchk( cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice) );

    int nthreads = PARTICLES_PER_BLOCK;
    int nblks = nparticles / nthreads + 1;
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_timestep_kernel<<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, m, q,  nparticles);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError(); 
    // CHECK_CUDA_ERROR(err);
    gpuErrchk( cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost) );


    // err = cudaGetLastError(); 
    // CHECK_CUDA_ERROR(err);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";

    
    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        // double t = atan2(y2, y1);
        // t += 2*M_PI*(t < 0);
        
        // last location in Boozer coordinates
        particle_output[7*i] = y1;
        particle_output[7*i + 1] = y2;
        particle_output[7*i + 2] = z;
        particle_output[7*i + 3] = v_par;
        particle_output[7*i + 4] = particles[i].t;
        // std::cout << "copied back t=" << particles[i].t << "\n";
        particle_output[7*i + 5] = particles[i].step_accept;
        particle_output[7*i + 6] = particles[i].step_attempt;
    }


    delete[] particles;
    gpuErrchk( cudaFree(particles_d) );
    return particle_output;
}