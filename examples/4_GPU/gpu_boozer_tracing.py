#!/usr/bin/env python
import pandas as pd

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from math import sqrt
from booz_xform import Booz_xform

from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec
from simsopt.util import in_github_actions
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )
filename = os.path.join('./examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')

import simsoptpp as sopp

from simsopt.util import boozer_interpolant
from simsopt.util import sample_stz
 
# # Compute the pdf of birth rate in s
# def s_density(s):
# 	return ((1-s**5)**2)*((1-s)**(-2/3))*np.exp(-19.94*(12*(1-s))**(-1/3))

# # Rejection sample s
# def sample_s():
# 	bound = 3e-4
# 	x = np.random.uniform()
# 	y = bound * np.random.uniform()

# 	while s_density(x) < y:
# 		assert s_density(x) <= bound
# 		x = np.random.uniform()
# 		y = bound * np.random.uniform()
# 	return x

# # Sample theta, zeta for a given s via rejection sampling
# def sample_tz(s, J_max, field):
# 	J = rand_J = 0
# 	while rand_J  >= J:
# 		theta = np.random.uniform(low=0, high=2*math.pi, size=1)
# 		zeta = np.random.uniform(low=0, high=2*math.pi, size=1)
# 		rand_J = np.random.uniform(low=0, high=J_max, size=1)

# 		loc = np.array([s, theta[0], zeta[0]]).reshape(1,3)
# 		field.set_points(loc)

# 		G = field.G()
# 		iota = field.iota()
# 		I = field.I()
# 		modB = field.modB()
# 		J = (G + iota*I)/(modB**2)
# 		J = J[0][0]
# 		assert J <= J_max
# 	return theta[0], zeta[0]

# # Sample s,t,z 
# def sample_stz(field, J_max):
# 	s = sample_s()
# 	theta, zeta = sample_tz(s, J_max, field)
# 	return np.array([s, theta, zeta])

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


# set seed for consistency
np.random.seed(8)

# trace particles
nparticles = 25000

stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
vpar = np.sqrt(2*ENERGY/MASS)

vpar_inits = vpar * np.random.uniform(low=-1, high=1, size=nparticles)

print("tracing particles")

# trace on GPU
last_time = sopp.boozer_gpu_tracing(
	quad_pts=quad_info, 
	srange=srange,
	trange=trange,
	zrange=zrange, 
	stz_init=stz_inits,
	m=MASS, 
	q=CHARGE, 
	vtotal=sqrt(2*ENERGY/MASS),  
	vtang=vpar_inits, 
	tmax=1e-4, 
	tol=1e-9, 
	psi0=field.psi0, 
	nparticles=nparticles)

last_time = np.reshape(last_time, (nparticles, 7))


particle_data = pd.DataFrame({'s_start': stz_inits[:,0], 't_start': stz_inits[:,1], 'z_start':stz_inits[:,2], 'vpar_start':vpar_inits,
							  's_end': last_time[:,0], 't_end':last_time[:,1], 'z_end':last_time[:,2], 'vpar_end':last_time[:,3], 'last_time':last_time[:,4],
							  'steps_accepted':last_time[:,5], 'steps_attempted':last_time[:,6]})
particle_data.to_csv('qh_particle_data.csv')


did_leave = [t < 1e-2 for t in particle_data['last_time']]
loss_frac = sum(did_leave) / len(did_leave)
print(f"Number of particles= {nparticles}")
print(f"Loss fraction: {loss_frac:.3f}")


