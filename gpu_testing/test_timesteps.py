import simsoptpp as sopp
from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, IterationStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec
import numpy as np
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )
import os
from test_interpolant import *

np.random.seed(1800)


def test_derivs(field, nfp, n_metagrid_pts, n_test_pts, verify=True):
        # generate test points
        s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
        t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        stz = np.hstack((s,t,z))

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))


        ### NEW INTERPOLANT
        srange, trange, zrange, quad_info = setup_interpolant(field, nfp, n_metagrid_pts)
        stz = np.ascontiguousarray(stz)

        psi0 =field.psi0

        print("calculating new derivatives")
        new_derivs = sopp.test_derivatives(quad_info, srange, trange, zrange, stz, vpar_init, VELOCITY, MASS, CHARGE, psi0, stz.shape[0])
        new_derivs = np.reshape(new_derivs, (stz.shape[0], 4))

        if verify:
                print("computing simsopt derivatives")
                old_derivs = np.empty((n_test_pts, 4))
                for i in range(n_test_pts):
                        old_derivs[i,:] = sopp.simsopt_derivs(field, stz[i,:], MASS, CHARGE, VELOCITY, vpar_init[i])


                rel_err = np.abs((old_derivs - new_derivs) / old_derivs)
                diff = np.max(rel_err)
                print(np.abs(old_derivs - new_derivs) / old_derivs)

                print("Maximum relative error in derivative values on {} points: {}".format(n_test_pts, diff))

                print("culprit particle:")
                row_index = np.argmax(rel_err) // rel_err.shape[1]
                print(stz[row_index, :])
                print(vpar_init[row_index])
                print("simsopt", old_derivs[row_index, :])
                print("new", new_derivs[row_index, :])
                print(rel_err[row_index, :])





def test_timestep(field, nfp, n_metagrid_pts, n_test_pts, verify=True):

        # generate test points
        s = np.random.uniform(low=0, high=0.95, size=(n_test_pts,1))
        t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        stz = np.hstack((s,t,z))

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

        

        print("computing new timesteps")
        srange, trange, zrange, quad_info = setup_interpolant(field, nfp, n_metagrid_pts)
        stz = np.ascontiguousarray(stz)
        psi0 = field.psi0
        last_time = sopp.test_timestep(
                quad_pts=quad_info, 
                srange=srange,
                trange=trange,
                zrange=zrange, 
                stz_init=stz,
                m=MASS, 
                q=CHARGE, 
                vtotal=np.sqrt(2*ENERGY/MASS),  
                vtang=vpar_init, 
                tol=1e-9, 
                psi0=psi0, 
                nparticles=n_test_pts)


        last_time = np.reshape(last_time, (n_test_pts, 7))

        new_final_positions = np.array([[x[4], x[0], x[1], x[2], x[3]] for x in last_time])

        if verify:
                print("computing simsopt timestep")

        
                gc_tys, gc_zeta_hits = trace_particles_boozer(
                        field, stz, vpar_init, tmax=1e-2, mass=MASS, charge=CHARGE,
                        Ekin=ENERGY, zetas=[0], tol=1e-9, stopping_criteria=[IterationStoppingCriterion(1)],
                        forget_exact_path=True)
                
                final_positions = np.array([x[-1] for x in gc_tys])
                rel_err = np.abs((final_positions - new_final_positions) / final_positions)
                diff = np.max(rel_err)
                print(np.abs(final_positions - new_final_positions) / final_positions)

                print("Maximum relative error in final positions on {} points: {}".format(n_test_pts, diff))

                print("culprit particle:")
                row_index = np.argmax(rel_err) // rel_err.shape[1]
                print(stz[row_index, :])
                print(vpar_init[row_index])
                print("simsopt", final_positions[row_index, :])
                print("new", new_final_positions[row_index, :])
                print(rel_err[row_index, :])


if __name__ == "__main__":
        n_metagrid_pts = 15


        # create a B-field
        filename = os.path.join('./examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')
        vmec = Vmec(filename)

        order = 3
        bri = BoozerRadialInterpolant(vmec, order, enforce_vacuum=True)

        nfp = vmec.wout.nfp
        degree = 3
        srange = (0, 1, n_metagrid_pts)
        thetarange = (0, np.pi, n_metagrid_pts)
        zetarange = (0, 2*np.pi/nfp, n_metagrid_pts)
        field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, extrapolate=True, nfp=nfp, stellsym=True)

        print("derivatives")
        test_derivs(field, nfp, 15, 100000)
        test_derivs(field, nfp, 15, 200000, False)
        test_derivs(field, nfp, 15, 400000, False)
        test_derivs(field, nfp, 15, 800000, False)
        test_derivs(field, nfp, 15, 1600000, False)
        test_derivs(field, nfp, 15, 3200000, False)
        test_derivs(field, nfp, 15, 6400000, False)
        test_derivs(field, nfp, 15, 12800000, False)
        test_derivs(field, nfp, 15, 25600000, False)
        test_derivs(field, nfp, 15, 51200000, False)
        test_derivs(field, nfp, 15, 102400000, False)

        print("time steps")
        test_timestep(field, nfp, 15, 100000)
        test_timestep(field, nfp, 15, 200000, False)
        test_timestep(field, nfp, 15, 400000, False)
        test_timestep(field, nfp, 15, 800000, False)
        test_timestep(field, nfp, 15, 1600000, False)
        test_timestep(field, nfp, 15, 3200000, False)
        test_timestep(field, nfp, 15, 6400000, False)
        test_timestep(field, nfp, 15, 12800000, False)


## Baseline
# Maximum relative error in derivative values on 100000 points: 9.939316856524188e-09, time = 1.39584ms
# 200,000 pts:      0.84192ms
# 400,000 pts:      1.62259ms
# 800,000 pts:      2.99683ms
# 1,600,000 pts:    5.99888ms
# 3,200,000 pts:    11.6332ms
# 6,400,000 pts:    23.1388ms
# 12,800,000 pts:   46.183ms
# 25,600,000 pts:   88.5732ms
# 51,200,000 pts:   165.522ms
# 102,400,000 pts:  314.623ms

# Maximum relative error in timesteps values on 100000 points: 2.6487780450361223e-07, time = 9.32342ms
# 200,000 pts:      19.2365ms
# 400,000 pts:      37.3183ms
# 800,000 pts:      73.4362ms
# 1,600,000 pts:    145.879ms
# 3,200,000 pts:    288.092ms
# 6,400,000 pts:    553.342ms
# 12,800,000 pts:   1114.85ms



## Remove in-bounds checks (no speed up)
# Maximum relative error in derivative values on 100000 points: 9.939316856524188e-09, time = 1.41965ms
# 200,000 pts:      0.896384ms
# 400,000 pts:      1.7072ms
# 800,000 pts:      3.21571ms
# 1,600,000 pts:    6.38931ms
# 3,200,000 pts:    12.2119ms
# 6,400,000 pts:    24.166ms
# 12,800,000 pts:   48.0481ms
# 25,600,000 pts:   96.1497ms
# 51,200,000 pts:   186.904ms
# 102,400,000 pts:  319.265ms

# Maximum relative error in final positions on 100000 points: 2.6487780450361223e-07, time = 9.1303ms
# 200,000 pts:      19.0297ms
# 400,000 pts:      36.6498ms
# 800,000 pts:      72.1688ms
# 1,600,000 pts:    143.237ms
# 3,200,000 pts:    285.249ms
# 6,400,000 pts:    566.618ms
# 12,800,000 pts:   1105.52ms


## Remove p.symm_exploited control flow
# Maximum relative error in derivative values on 100000 points: 9.939316856524188e-09, time = 1.99347ms
# 200,000 pts:      0.890848ms
# 400,000 pts:      1.71552ms
# 800,000 pts:      3.2305ms
# 1,600,000 pts:    6.38234ms
# 3,200,000 pts:    12.2689ms
# 6,400,000 pts:    24.2622ms
# 12,800,000 pts:   48.0764ms
# 25,600,000 pts:   92.4119ms
# 51,200,000 pts:   175.952ms
# 102,400,000 pts:  323.464ms

# Maximum relative error in final positions on 100000 points: 2.6487780450361223e-07, time = 9.3864ms
# 200,000 pts:      18.9911ms
# 400,000 pts:      36.7144ms
# 800,000 pts:      72.3696ms
# 1,600,000 pts:    143.615ms
# 3,200,000 pts:    285.735ms
# 6,400,000 pts:    548.519ms
# 12,800,000 pts:   1094.67ms


# Remove redundant arithmetic in the interpolant (small speed up for largest input)
# Maximum relative error in derivative values on 100000 points: 9.939316856524188e-09, time = 1.46317ms
# 200,000 pts:      0.887104ms
# 400,000 pts:      1.71107ms
# 800,000 pts:      3.21958ms
# 1,600,000 pts:    6.31952ms
# 3,200,000 pts:    12.2546ms
# 6,400,000 pts:    24.1625ms
# 12,800,000 pts:   47.9874ms
# 25,600,000 pts:   96.1531ms
# 51,200,000 pts:   184.246ms
# 102,400,000 pts:  338.85ms

# Maximum relative error in final positions on 100000 points: 2.6487780450361223e-07, time = 9.26304ms
# 200,000 pts:      18.9922ms
# 400,000 pts:      36.5945ms
# 800,000 pts:      72.2518ms
# 1,600,000 pts:    143.572ms
# 3,200,000 pts:    285.856ms
# 6,400,000 pts:    551.471ms
# 12,800,000 pts:   1093.59ms