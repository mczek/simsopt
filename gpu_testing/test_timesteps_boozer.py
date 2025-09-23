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
from simsopt.util import boozer_interpolant
np.random.seed(1800)


def test_derivs(field, nfp, n_metagrid_pts, n_test_pts):
        # generate test points
        s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
        t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        stz = np.hstack((s,t,z))

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

        # print("computing simsopt derivatives")
        old_derivs = np.empty((n_test_pts, 4))
        for i in range(n_test_pts):
                old_derivs[i,:] = sopp.simsopt_derivs_boozer(field, stz[i,:], MASS, CHARGE, VELOCITY, vpar_init[i])


        ### NEW INTERPOLANT
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, n_metagrid_pts)
        stz = np.ascontiguousarray(stz)

        psi0 =field.psi0

        # print("calculating new derivatives")
        new_derivs = sopp.test_derivatives_boozer(quad_info, srange, trange, zrange, stz, vpar_init, VELOCITY, MASS, CHARGE, psi0, stz.shape[0])
        new_derivs = np.reshape(new_derivs, (stz.shape[0], 4))

        rel_err = np.abs((old_derivs - new_derivs) / old_derivs)
        diff = np.max(rel_err)
        # print(np.abs(old_derivs - new_derivs) / old_derivs)

        print("Maximum relative error in derivative values on {} points: {}".format(n_test_pts, diff))

        if diff > 1e-8:
                print("BOOZER RHS TEST FAILED")

                print("culprit particle:")
                row_index = np.argmax(rel_err) // rel_err.shape[1]
                print(stz[row_index, :])
                print(vpar_init[row_index])
                print("simsopt", old_derivs[row_index, :])
                print("new", new_derivs[row_index, :])
                print(rel_err[row_index, :])
        else:
                print("BOOZER RHS TEST SUCCESS")





def test_timestep(field, nfp, n_metagrid_pts, n_test_pts):

        # generate test points
        s = np.random.uniform(low=0, high=0.95, size=(n_test_pts,1))
        t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        stz = np.hstack((s,t,z))

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

        # print("computing simsopt timestep")

        
        gc_tys, gc_zeta_hits = trace_particles_boozer(
                field, stz, vpar_init, tmax=1e-2, mass=MASS, charge=CHARGE,
                Ekin=ENERGY, zetas=[0], tol=1e-9, stopping_criteria=[IterationStoppingCriterion(1)],
                forget_exact_path=True)
        
        final_positions = np.array([x[-1] for x in gc_tys])

        # print("computing new timesteps")
        srange, trange, zrange, quad_info, maxJ = boozer_interpolant(field, nfp, n_metagrid_pts)
        stz = np.ascontiguousarray(stz)
        psi0 = field.psi0
        last_time = sopp.test_timestep_boozer(
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

        rel_err = np.abs((final_positions - new_final_positions) / final_positions)
        diff = np.max(rel_err)
        # print(np.abs(final_positions - new_final_positions) / final_positions)

        print("Maximum relative error in final positions on {} points: {}".format(n_test_pts, diff))
        if diff > 1e-8:
                print("BOOZER TIMESTEP TEST FAILED")

                print("culprit particle:")
                row_index = np.argmax(rel_err) // rel_err.shape[1]
                print(stz[row_index, :])
                print(vpar_init[row_index])
                print("simsopt", final_positions[row_index, :])
                print("new", new_final_positions[row_index, :])
                print(rel_err[row_index, :])
        else:
                print("BOOZER TIMESTEP TEST SUCCESS")
        


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

        test_derivs(field, nfp, 15, 10000)

        test_timestep(field, nfp, 15, 10000)