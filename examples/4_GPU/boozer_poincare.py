import pandas as pd

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from math import sqrt
from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec
from simsopt.util import in_github_actions
from simsopt.util.constants import (
        PROTON_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ONE_EV,
        ELEMENTARY_CHARGE as CHARGE
        )

import simsoptpp as sopp

from simsopt.util import boozer_interpolant
from simsopt.util import sample_stz

np.random.seed(1865)
 

### Set up a Boozer field
filename = os.path.join('./examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')

# Compute VMEC equilibrium
vmec = Vmec(filename)

# Construct radial interpolant of magnetic field
order = 3
bri = BoozerRadialInterpolant(vmec, order, enforce_vacuum=True)

# Construct 3D interpolation
nfp = vmec.wout.nfp
degree = 3
srange = (0, 1, 15)
thetarange = (0, np.pi, 15)
zetarange = (0, 2*np.pi/nfp, 15)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

### Maximum tracing time
tmax = 1e-3

### Sample initial conditions
s_vals = np.linspace(0, 1, 10, dtype=np.float64)
theta_vals = np.linspace(-np.pi, np.pi, 10, dtype=np.float64)
stz = np.vstack([[s_val, theta_val, 0] for s_val in s_vals for theta_val in theta_vals], dtype=np.float64)

nparticles = stz.shape[0]
vpar_init = np.sqrt(2*1e3*ONE_EV/MASS)* 0.5*np.ones(nparticles, dtype=np.float64)#np.random.uniform(-1, 1, nparticles)


### First, get poincare data from simsopt



# trace this particle in simsopt, saving the trajectory
# print("simsopt tracing...")
# gc_tys, gc_zeta_hits = trace_particles_boozer(
#     field, stz, vpar_init, tmax=tmax, mass=MASS, charge=CHARGE,
#     Ekin=1e3*ONE_EV, zetas=[0], tol=1e-9, stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.99)],
#     forget_exact_path=True)

# # print(gc_zeta_hits)
# rows = []
# for i in range(len(gc_zeta_hits)):
#     if len(gc_zeta_hits[i]) > 0:
#         # print(gc_zeta_hits[i])
#         loc = gc_zeta_hits[i][:, 2:5]
#         n_hits = loc.shape[0]
#         rows += [np.hstack((loc, i*np.ones((n_hits,1))))]
# full_data = np.vstack(rows)
# simsopt_poincare_data = pd.DataFrame({'x1': full_data[:, 0], 'x2': full_data[:, 1], 'zeta': full_data[:, 2], "id":full_data[:,3]})
# simsopt_poincare_data['s'] = np.sqrt(simsopt_poincare_data['x1']**2 + simsopt_poincare_data['x2']**2)
# simsopt_poincare_data['theta'] = np.arctan2(simsopt_poincare_data['x2'], simsopt_poincare_data['x1'])
# simsopt_poincare_data.to_csv("examples/4_GPU/cpu_boozer_poincare.csv")

# exit()

### Now create poincare data with GPU tracing
### using bisection algorithm

# create gpu interpolant data

print("generating b field interpolant data")
srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, 15)

# First set an initial distance at which to save
dt_save = 1e-8

gpu_stz = stz.copy()
gpu_vpar = vpar_init.copy()

gpu_time = 0
snapshots = []
ids = list(range(nparticles))
while gpu_time < tmax:
    print(gpu_time)
    gpu_time += dt_save

    # print("before", gpu_stz)
    last_time = sopp.boozer_gpu_tracing(
        quad_pts=quad_info, 
        srange=srange,
        trange=trange,
        zrange=zrange, 
        stz_init=gpu_stz.copy(),
        m=MASS, 
        q=CHARGE, 
        vtotal=np.sqrt(2*1e3*ONE_EV/MASS),  
        vtang=gpu_vpar.copy(), 
        tmax=dt_save, 
        tol=1e-9, 
        psi0=field.psi0, 
        nparticles=len(ids))
    # print("after", gpu_stz)


    last_time = np.reshape(last_time, (len(ids), 7))

    # save snapshot s, theta, zeta, vpar
    snapshot_data = pd.DataFrame({'s':last_time[:, 0],
                                  'theta':last_time[:,1],
                                  'zeta':last_time[:,2],
                                  'vpar':last_time[:,3]})
    snapshot_data['time'] = gpu_time
    snapshot_data['id'] = ids
    snapshots += [snapshot_data]

    # remove lost particles
    ids = [ids[i] for i in range(len(ids)) if last_time[i,0] < 1]
    last_time = last_time[last_time[:,0] < 1.0] # s < 1

    # update particle locations
    gpu_stz = last_time[:, 0:3]
    gpu_vpar = last_time[:, 3]
    gpu_stz = np.ascontiguousarray(gpu_stz, dtype=np.float64)
    gpu_vpar = np.ascontiguousarray(gpu_vpar, dtype=np.float64)


full_snapshots = pd.concat(snapshots, ignore_index=True)
full_snapshots = full_snapshots.sort_values(by = ['id', 'time'])
full_snapshots['zeta_mod_2pi'] = np.mod(full_snapshots['zeta'], 2*np.pi)
full_snapshots.to_csv("examples/4_GPU/boozer_gpu_snapshots.csv")

# check for poincare punctures
def find_punctures(df, tol=1e-3):
    print("finding punctures...")
    df = df.sort_values(by = ['id', 'time'])

    print("sorted df")
    print(df)
    ids = df['id']
    zetas = np.mod(df['zeta'], 2*np.pi)
    zetas_shift = zetas - np.pi
    nrows = df.shape[0]
    # print(zetas)
    # print(np.min(zetas))
    # print(np.max(zetas))
    # print(zetas_shift)

    # satisfy a tolerance
    diff = np.minimum(np.abs(zetas), np.abs(zetas - 2*np.pi))

    zeta_crossing = [(np.abs(zetas[i]) < np.pi/2) and (np.sign(zetas_shift[i]) != np.sign(zetas_shift[i+1])) for i in range(nrows-1)]
    no_match = [(diff[i] > tol) and (diff[i+1] > tol) for i in range(nrows-1)]
    need_refine = [(ids[i] == ids[i+1]) and zeta_crossing[i] and no_match[i] for i in range(nrows-1)]

    return need_refine + [False] # to use as a mask


need_refinement = find_punctures(full_snapshots)
while(np.sum(need_refinement) > 0):
    dt_save /= 2

    print("Refining, new dt_save = ", dt_save)
    print("Number needing refinement: ")
    print(np.sum(need_refinement))
    # exit()
    # print(need_refinement)
    # exit()
    next_starts = full_snapshots[need_refinement]
    print(next_starts)
    # exit()
    refine_stz = next_starts[['s', 'theta', 'zeta']].to_numpy()
    refine_vpar = next_starts['vpar'].to_numpy()
    refine_times = next_starts['time']
    # exit()
    nparticles = refine_stz.shape[0]
    last_time = sopp.boozer_gpu_tracing(
        quad_pts=quad_info, 
        srange=srange,
        trange=trange,
        zrange=zrange, 
        stz_init=refine_stz.copy(),
        m=MASS, 
        q=CHARGE, 
        vtotal=np.sqrt(2*1e3*ONE_EV/MASS),  
        vtang=refine_vpar.copy(), 
        tmax=dt_save, 
        tol=1e-9, 
        psi0=field.psi0, 
        nparticles=nparticles)
    last_time = np.reshape(last_time, (nparticles, 7))
    # exit()
    # print(refine_times)
    # print(last_time)
    snapshot_data = pd.DataFrame({'s':last_time[:, 0],
                                    'theta':last_time[:,1],
                                    'zeta':last_time[:,2],
                                    'vpar':last_time[:,3],
                                    'time':refine_times + dt_save})
    print("new locations", snapshot_data)
    snapshot_data['time'] = gpu_time
    snapshot_data['id'] = next_starts['id']
    full_snapshots = pd.concat([full_snapshots, snapshot_data], ignore_index=True)
    need_refinement = find_punctures(full_snapshots)

zetas = np.mod(full_snapshots['zeta'], 2*np.pi)
diff = np.minimum(np.abs(zetas), np.abs(zetas - 2*np.pi))
poincare_hits = full_snapshots[diff < 1e-3]
poincare_hits.to_csv("examples/4_GPU/boozer_gpu_poincare.csv")
full_snapshots.to_csv("examples/4_GPU/boozer_gpu_full.csv")
exit()
# print(full_snapshots)
# print(need_refine) 
# print(np.sum(zeta_crossing))
# print(np.sum(need_refine))
exit()

gpu_stz = stz.copy()
cpu_vpar = vpar_init.copy()

gpu_stz = stz.copy()
gpu_vpar = vpar_init.copy()

tracing_time = 0
snapshots = []
while tracing_time < tmax:
    print(tracing_time)

    cpu_final_stz = np.vstack([gc_tys[i][-1][1:4] for i in range(nparticles)])
    cpu_final_vpar = np.vstack([gc_tys[i][-1][4] for i in range(nparticles)])
    cpu_final_t = np.vstack([gc_tys[i][:, 0] for i in range(nparticles)])



    snapshot_data = pd.DataFrame({'s_start' : stz[:,0], "theta_start": stz[:, 1], "zeta_start":stz[:,2], 'vpar_start':vpar_init,
                                'cpu_s_end': cpu_final_stz[:, 0], 'cpu_theta_end':cpu_final_stz[:, 1], 'cpu_zeta_end':cpu_final_stz[:, 2], 'cpu_vpar_end': cpu_final_vpar[:, 0], 'cpu_final_t':cpu_final_t[:, 0],
                                'gpu_s_end' : last_time[:, 0], 'gpu_theta_end':last_time[:, 1], 'gpu_zeta_end':last_time[:,2], 'gpu_vpar_end':last_time[:,3]})
    snapshot_data['time'] = tracing_time
    snapshot_data['id'] = list(range(nparticles))
    snapshots += [snapshot_data]
    tracing_time += dt_save

    cpu_stz = cpu_final_stz
    cpu_vpar = cpu_final_vpar

    for i in range(3):
        gpu_stz[:, i] = last_time[:, i]
    gpu_vpar = last_time[:, 3]

output_data = pd.concat(snapshots, ignore_index=True)
output_data['s_rel_err'] = np.abs(output_data['cpu_s_end'] - output_data['gpu_s_end']) / output_data['cpu_s_end']
output_data['theta_rel_err'] = np.abs(output_data['cpu_theta_end'] - output_data['gpu_theta_end']) / output_data['cpu_theta_end']
output_data['zeta_rel_err'] = np.abs(output_data['cpu_zeta_end'] - output_data['gpu_zeta_end']) / output_data['cpu_zeta_end']
output_data['vpar_rel_err'] = np.abs(output_data['cpu_vpar_end'] - output_data['gpu_vpar_end']) / output_data['cpu_vpar_end']

output_data['cpu_x1'] = output_data['cpu_s_end']*np.cos(output_data['cpu_theta_end'])
output_data['cpu_x2'] = output_data['cpu_s_end']*np.sin(output_data['cpu_theta_end'])
output_data['gpu_x1'] = output_data['gpu_s_end']*np.cos(output_data['gpu_theta_end'])
output_data['gpu_x2'] = output_data['gpu_s_end']*np.sin(output_data['gpu_theta_end'])

output_data['x1_diff'] = output_data['cpu_x1'] - output_data['gpu_x1']
output_data['x2_diff'] = output_data['cpu_x2'] - output_data['gpu_x2']
output_data['zeta_diff'] = output_data['cpu_zeta_end'] - output_data['gpu_zeta_end']
output_data['dist'] = np.sqrt(output_data['x1_diff']**2 + output_data['x2_diff']**2 + output_data['zeta_diff']**2)

output_data.to_csv("./boozer_device_comparison.csv")
