# This file is for comparing a particle's trajectory in the cpu vs gpu code
# It is intended for debugging purposes

#!/usr/bin/env python
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
 

# Set up a Boozer field
filename = os.path.join('./examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')

# Compute VMEC equilibrium
t1 = time.time()
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

srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, 15)

# pick tmax and a single particle
tmax = 1e-4
dt_save = 1e-6
nparticles = 1000
stz = np.vstack([sample_stz(field, maxJ) for _ in range(nparticles)])
vpar_init = np.sqrt(2*1e3*ONE_EV/MASS)*np.random.uniform(-1, 1, nparticles)

cpu_stz = stz.copy()
cpu_vpar = vpar_init.copy()

gpu_stz = stz.copy()
gpu_vpar = vpar_init.copy()

tracing_time = 0
snapshots = []
while tracing_time < tmax:
    print(tracing_time)

    # trace this particle in simsopt, saving the trajectory
    gc_tys, gc_zeta_hits = trace_particles_boozer(
        field, cpu_stz, cpu_vpar, tmax=dt_save, mass=MASS, charge=CHARGE,
        Ekin=1e3*ONE_EV, tol=1e-9, stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.99)],
        forget_exact_path=True)

    cpu_final_stz = np.vstack([gc_tys[i][-1][1:4] for i in range(nparticles)])
    cpu_final_vpar = np.vstack([gc_tys[i][-1][4] for i in range(nparticles)])
    cpu_final_t = np.vstack([gc_tys[i][:, 0] for i in range(nparticles)])

    last_time = sopp.boozer_gpu_tracing(
        quad_pts=quad_info, 
        srange=srange,
        trange=trange,
        zrange=zrange, 
        stz_init=gpu_stz.copy(),
        m=MASS, 
        q=CHARGE, 
        vtotal=np.sqrt(2*1e3*ONE_EV/MASS),  
        vtang=gpu_vpar, 
        tmax=dt_save, 
        tol=1e-9, 
        psi0=field.psi0, 
        nparticles=nparticles)

    last_time = np.reshape(last_time, (nparticles, 7))


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
