#!/usr/bin/env python
import pandas as pd

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from math import sqrt

from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec
from simsopt.util import in_github_actions
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV

filename = os.path.join('/global/homes/m/mczek/simsopt/examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')
# filename = '/global/homes/m/mczek/simsopt/input.misha'
print(filename)
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')

import simsoptpp as sopp

"""
Here we trace particles in the vacuum magnetic field in Boozer coordinates
from the QH equilibrium in Landreman & Paul (arXiv:2108.03711). We evaluate
energy conservation and compute resonant particle trajectories.
"""

# Sample s
def s_density(s):
	return ((1-s**5)**2)*((1-s)**(-2/3))*np.exp(-19.94*(12*(1-s))**(-1/3))

def sample_s():
	bound = 3e-4
	x = np.random.uniform()
	y = bound * np.random.uniform()

	while s_density(x) < y:
		assert s_density(x) <= bound
		x = np.random.uniform()
		y = bound * np.random.uniform()
	return x

# Sample theta, zeta
def sample_tz(s, J_max, field):
	J = rand_J = 0
	while rand_J  >= J:
		theta = np.random.uniform(low=0, high=2*math.pi, size=1)
		zeta = np.random.uniform(low=0, high=2*math.pi, size=1)
		rand_J = np.random.uniform(low=0, high=J_max, size=1)
	
		#print(s, theta, zeta)
		loc = np.array([s, theta[0], zeta[0]]).reshape(1,3)
		field.set_points(loc)

		G = field.G()
		iota = field.iota()
		I = field.I()
		modB = field.modB()
		J = (G + iota*I)/(modB**2)
		J = J[0][0]
		assert J <= J_max
	return theta[0], zeta[0]

def sample_stz(field, J_max):
	s = sample_s()
	
	theta, zeta = sample_tz(s, J_max, field)
	return np.array([s, theta, zeta])

def particle_path_df(path, id, tmax):
	df = pd.DataFrame(path)
	df.columns = ['time', 's', 'theta', 'zeta', 'v_par']
	df['id'] = id
	df['lost'] = path[-1][0] < tmax - 1e-15
	df['theta'] = df['theta'] % 2*math.pi
	df['zeta'] = df['zeta'] % 2*math.pi
	df = df[df['time'] == 0]
	return df
	




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

# Evaluate error in interpolation

print('Error in |B| interpolation', field.estimate_error_modB(1000), flush=True)


# # Initialize vpar assuming mu = 0
Ekin = 5000*ONE_EV
mass = PROTON_MASS
vpar = np.sqrt(2*0.8*Ekin/mass)

# print(vpar_inits)
# gc_tys, gc_zeta_hits = trace_particles_boozer(
#     field, stz_inits, vpar_inits, tmax=1e-2, mass=mass, charge=ELEMENTARY_CHARGE,
#     Ekin=Ekin, tol=1e-8, mode='gc_vac', stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.99), MinToroidalFluxStoppingCriterion(0.01), ToroidalTransitStoppingCriterion(100, True)],
#     forget_exact_path=False)



	
t2 = time.time()

# CALCULATE MAX J
print("calculating J max")
n_grid_pts = 15
s_max = 1

srange = (0, s_max, n_grid_pts)
trange = (0, 2*np.pi, 3*n_grid_pts)
zrange = (0, 2*np.pi, 3*n_grid_pts)

s_grid = np.linspace(srange[0], srange[1], srange[2])
theta_grid = np.linspace(trange[0], trange[1], trange[2], endpoint=False)
zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2], endpoint=False)


print("building quad_pts")
grid_start = time.time()
quad_pts = np.empty((srange[2]*trange[2]*zrange[2], 3))
for i in range(srange[2]):
	for j in range(trange[2]):
		for k in range(zrange[2]):
			quad_pts[trange[2]*zrange[2]*i + zrange[2]*j + k, :] = [s_grid[i], theta_grid[j], zeta_grid[k]]
grid_end = time.time()
print("building grid time=", grid_end - grid_start)
print(quad_pts.shape)

print("building interpolation info")
interp_start = time.time()
field.set_points(quad_pts)
G = field.G()
iota = field.iota()
I = field.I()
modB = field.modB()
J = (G + iota*I)/(modB**2)
# minJ = np.min(J)
maxJ = np.max(J)
print("maxJ", maxJ)

psi0 = field.psi0

# Quantities to interpolate
print("interpolation points")
modB_derivs = field.modB_derivs()
interp_end = time.time()
print("interpolation time = ", interp_end-interp_start)

quad_info = np.hstack((modB, modB_derivs, G, iota))
quad_info = np.ascontiguousarray(quad_info)

# set seed
np.random.seed(8)


# TRACE NAIVE PARTICLES
t3 = time.time()
nparticles = 5000
filename = "boozer_tracing_loss_time.csv"

stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
vpar_inits = vpar * np.random.uniform(low=-1, high=1, size=nparticles)

print("tracing particles")


last_time = sopp.gpu_tracing(
	quad_pts=quad_info, 
	srange=srange,
	trange=trange,
	zrange=zrange, 
	stz_init=stz_inits,
	m=mass, 
	q=ELEMENTARY_CHARGE, 
	vtotal=sqrt(2*Ekin/mass),  
	vtang=vpar_inits, 
	tmax=1e-1, 
	tol=1e-9, 
	psi0=psi0, 
	nparticles=nparticles)



t4 = time.time()

did_leave = [t < 1e-2 for t in last_time]
loss_frac = sum(did_leave) / len(did_leave)
print(f"Number of particles= {nparticles}")
print(f"Loss fraction: {loss_frac:.3f}")
print(f"Total time for particle tracing={t4-t1:.3f}s.")
print(f"VMEC+simsopt interpolant setup time={t2-t1:.3f}s.")
print(f"generating interpolation points for tracing time={t3-t2:.3f}s.")
print(f"gpu_tracing function time for particle tracing={t4-t3:.3f}s.")




df = pd.DataFrame(stz_inits)
df.columns = ["s", "theta", "zeta"]
df['v_par'] = vpar_inits
df['last_time'] = last_time
df.to_csv(filename, header='column_names')