import simsoptpp as sopp
from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, IterationStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec
import numpy as np
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )
from simsopt.field.sampling import draw_uniform_on_surface
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, curves_to_vtk, BoozerSurface, Volume
from simsopt.util import in_github_actions, proc0_print

import os
import pandas as pd
from test_interpolant import *

np.random.seed(1800)
tmax = 1e-4
nparticles = 10000
n_metagrid_pts = 30

### CREATE A FIELD FOR TRACING
ID = 1655332
fID = ID // 1000
[surfaces, ma, coils] = load("./serial1655332.json")
nfp = surfaces[0].nfp
nc_per_hp = len(coils)//nfp//2
base_coils = coils[:nc_per_hp]
base_curves = [c.curve for c in base_coils]

# compute half radius surface
mpol = surfaces[0].mpol
ntor = surfaces[0].ntor
stellsym = surfaces[0].stellsym
nfp = surfaces[0].nfp
phis = np.linspace(0, 1/nfp, 2*mpol+1, endpoint=False)
thetas=np.linspace(0, 1, 2*ntor+1, endpoint=False)
surface_hr = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas, stellsym=stellsym)
surface_hr.x = surfaces[0].x
mR = surfaces[-1].minor_radius()
bs_bs = BiotSavart(coils)
label = Volume(surface_hr)
targetlabel = np.sign(surface_hr.volume()) * 2*np.pi * 1.0 * (np.pi * (mR/2)**2)
boozer_surface = BoozerSurface(bs_bs, surface_hr, label, targetlabel)

# df = pd.read_pickle('/mnt/home/mczekanski/ceph/QUASR_08072024/QUASR_08072024.pkl')
# iota0 = df[df.ID==ID].iloc[0].mean_iota
iota0 = 1.2
print("iota = ", iota0)
coil_currents = [c.current.get_value() for c in coils]
G0 = 2. * np.pi * np.sum(np.abs(coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=20, iota=iota0, G=G0)

if res['success']:
    proc0_print('HALF RADIUS COMPUTATION SUCCESS', flush=True)
else:
    proc0_print('HALF RADIUS COMPUTATION FAILURE', flush=True)
    quit()

# RESCALE TO ARES-CS MINOR RADIUS
mR = surfaces[-1].minor_radius()
scale = 1.7 / mR
for s in surfaces + [surface_hr]:
    s.x = scale*s.x
for c in base_curves:
    c.x = scale*c.x
# ma.x = scale * ma.x

# MAKE THE MAGNETIC FIELD ON AXIS 5.685257882303897
bs = BiotSavart(coils)
meanB = np.mean(bs.set_points(ma.gamma()).AbsB())

scale = 5.685257882303897/meanB
for c in base_coils:
    c.current.x = scale*c.current.x


curves = [c.curve for c in coils]
s_hp = surfaces[-1]
s_outer = SurfaceXYZTensorFourier(mpol=s_hp.mpol, ntor=s_hp.ntor, stellsym=s_hp.stellsym, nfp=nfp,\
        quadpoints_phi=np.linspace(0, 1, 256, endpoint=False), quadpoints_theta=np.linspace(0, 1, 256, endpoint=False))
s_outer.x = s_hp.x

# create loss surface
sc_particle = SurfaceClassifier(s_outer, h=0.1, p=2)

# sample initial conditions
rs = np.linalg.norm(s_outer.gamma()[:, :, 0:2], axis=2)
zs = s_outer.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n_metagrid_pts)
phirange = (0, 2*np.pi/nfp, n_metagrid_pts)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, np.max(zs), n_metagrid_pts)
print(rrange, phirange, zrange)
field = InterpolatedField(
        bs, 3, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
)


### SAMPLE ICs
s_spawn_coarse = boozer_surface.surface
s_spawn_fine = SurfaceXYZTensorFourier(mpol=s_spawn_coarse.mpol, ntor=s_spawn_coarse.ntor, \
        stellsym=s_spawn_coarse.stellsym, nfp=s_spawn_coarse.nfp,\
        quadpoints_phi=np.linspace(0, 1/s_spawn_coarse.nfp, 1000, endpoint=False), \
        quadpoints_theta=np.linspace(0, 1, 1000, endpoint=False))
s_spawn_fine.x = s_spawn_coarse.x

xyz_init, _ = draw_uniform_on_surface(s_spawn_fine, nparticles, safetyfactor=10)

VELOCITY = np.sqrt(2 * ENERGY / MASS)
vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (nparticles,))

xyz_init = np.ascontiguousarray(xyz_init)
vpar_init = np.ascontiguousarray(vpar_init)


### SIMSOPT TRACING
# print("simsopt tracing")
# gc_tys, gc_phi_hits = trace_particles(
#         field, xyz_init, vpar_init, tmax=tmax, mass=MASS, charge=CHARGE,
#         Ekin=ENERGY, tol=1e-9, phis=[],
#         stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode='gc_vac', forget_exact_path=True,
#         phase_angle=0)

# cpu_last_time = [trajectory[-1][0] for trajectory in gc_tys]
# cpu_last_x = [trajectory[-1][1] for trajectory in gc_tys]
# cpu_last_y = [trajectory[-1][2] for trajectory in gc_tys]
# cpu_last_z = [trajectory[-1][3] for trajectory in gc_tys]
# cpu_last_vpar = [trajectory[-1][4] for trajectory in gc_tys]



### SETUP NEW INTERPOLANT
r_range = (field.r_range[0], field.r_range[1], 3*field.r_range[2]+1)
phi_range = (field.phi_range[0], field.phi_range[1], 3*field.phi_range[2]+1)
z_range = (field.z_range[0], field.z_range[1], 3*field.z_range[2]+1)

r_grid = np.linspace(r_range[0], r_range[1], r_range[2])
phi_grid = np.linspace(phi_range[0], phi_range[1], phi_range[2])
z_grid = np.linspace(z_range[0], z_range[1], z_range[2])

quad_pts = np.empty((r_range[2]*phi_range[2]*z_range[2], 3))
for i in range(r_range[2]):
    for j in range(phi_range[2]):
        for k in range(z_range[2]):
            quad_pts[phi_range[2]*z_range[2]*i + z_range[2]*j + k, :] = [r_grid[i], phi_grid[j], z_grid[k]]



field.set_points_cyl(quad_pts)

# Quantities to interpolate
B = field.B_cyl()
GradAbsB = field.GradAbsB_cyl()
    
signed_dist_vals = sc_particle.evaluate_rphiz(quad_pts)

quad_info = np.hstack((B, GradAbsB, signed_dist_vals))

# reorder for device memory accesses
print("reordering interpolant data form device accesses")
cell_quad_pts = np.empty((field.r_range[2]*field.z_range[2]*field.phi_range[2]*64, quad_info.shape[1]))
for cell_r in range(field.r_range[2]):
    for cell_phi in range(field.phi_range[2]):
        for cell_z in range(field.z_range[2]):
            row_start = 64*(cell_r*field.phi_range[2]*field.z_range[2] + cell_phi*field.z_range[2] + cell_z)

            # if cell_r == 24 and cell_phi == 22 and cell_z == 20:
            #     print(row_start)

            assert 3*cell_r + i < r_range[2]
            # iterate over spline locations for this cell
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        row_idx = row_start + 16*i + 4*j + k

                        # if cell_r == 24 and cell_phi == 22 and cell_z == 20:
                        #     print(row_idx)
                        # print(3*cell_r+i, 3*cell_phi+j, 3*cell_z+k)
                        # print(field.r_range[2], field.phi_range[2], field.z_range[2])
                        # print(row_idx, phi_range[2]*z_range[2]*(3*cell_r + i) + z_range[2]*(3*cell_phi+j) + 3*cell_z + k)
                        cell_quad_pts[row_idx,: ] = quad_info[phi_range[2]*z_range[2]*(3*cell_r + i) + z_range[2]*(3*cell_phi+j) + 3*cell_z + k, :]

quad_info = np.ascontiguousarray(cell_quad_pts)
### GPU TRACING
print("gpu tracing")
last_time = sopp.gpu_tracing(quad_info, r_range, phi_range, z_range, xyz_init, MASS, CHARGE, VELOCITY, vpar_init, tmax, 1e-9, nparticles)


last_time = np.reshape(last_time, (nparticles, 7))


# particle_data = pd.DataFrame({'x_start': xyz_init[:,0], 'y_start': xyz_init[:,1], 'z_start':xyz_init[:,2], 'vpar_start':vpar_init,
# 							  'gpu_x_end': last_time[:,0], 'cpu_x_end':cpu_last_x,
#                               'gpu_y_end':last_time[:,1], 'cpu_y_end':cpu_last_y,
#                               'gpu_z_end':last_time[:,2], 'cpu_z_end':cpu_last_z,
#                               'gpu_vpar_end':last_time[:,3], 'cpu_vpar_end':cpu_last_vpar,
#                               'gpu_last_time':last_time[:,4], 'cpu_last_time' : cpu_last_time,
# 							  'gpu_steps_accepted':last_time[:,5], 'gpu_steps_attempted':last_time[:,6]})
particle_data = pd.DataFrame({'x_start': xyz_init[:,0], 'y_start': xyz_init[:,1], 'z_start':xyz_init[:,2], 'vpar_start':vpar_init,
							  'gpu_x_end': last_time[:,0],
                              'gpu_y_end':last_time[:,1],
                              'gpu_z_end':last_time[:,2],
                              'gpu_vpar_end':last_time[:,3],
                              'gpu_last_time':last_time[:,4],
							  'gpu_steps_accepted':last_time[:,5], 'gpu_steps_attempted':last_time[:,6]})
particle_data.to_csv('cartesian_particle_data.csv')
