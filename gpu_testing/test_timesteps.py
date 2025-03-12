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

np.random.seed(1800)


def test_derivs(n_metagrid_pts):

        n_metagrid_pts = 15


        # create a B-field
        filename = os.path.join('/global/homes/m/mczek/simsopt/examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')
        vmec = Vmec(filename)

        order = 3
        bri = BoozerRadialInterpolant(vmec, order, enforce_vacuum=True)

        nfp = vmec.wout.nfp
        degree = 3
        srange = (0, 1, n_metagrid_pts)
        thetarange = (0, np.pi, n_metagrid_pts)
        zetarange = (0, 2*np.pi/nfp, n_metagrid_pts)
        field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, extrapolate=True, nfp=nfp, stellsym=True)

        # generate test points
        n_test_pts = 10000
        s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
        t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        stz = np.hstack((s,t,z))

        # stz = np.array([[1.00850907, 3.67470486, 3.96141234]])
        # print(stz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

        # vpar_init = np.array([6512489.775668796])

        print("computing simsopt derivatives")

        # SIMSOPT INTERPOLANT
        # print(stz)
        # field.set_points(stz)
        # G = field.G()
        # iota = field.iota()
        # modB = field.modB()
        # modB_derivs = field.modB_derivs()
        # simsopt_interpolation = np.hstack((modB, modB_derivs, G, iota))
        # print(simsopt_interpolation)

        old_derivs = np.empty((n_test_pts, 4))
        for i in range(n_test_pts):
                old_derivs[i,:] = sopp.simsopt_derivs(field, stz[i,:], MASS, CHARGE, VELOCITY, vpar_init[i])


        ### NEW INTERPOLANT
        srange = (0, 1, 3*n_metagrid_pts+1)
        trange = (0, np.pi, 3*n_metagrid_pts+1)
        zrange = (0, 2*np.pi/nfp, 3*n_metagrid_pts+1)

        s_grid = np.linspace(srange[0], srange[1], srange[2])
        theta_grid = np.linspace(trange[0], trange[1], trange[2])
        zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

        quad_pts = np.empty((srange[2]*trange[2]*zrange[2], 3))
        for i in range(srange[2]):
                for j in range(trange[2]):
                        for k in range(zrange[2]):
                                quad_pts[trange[2]*zrange[2]*i + zrange[2]*j + k, :] = [s_grid[i], theta_grid[j], zeta_grid[k]]

        field.set_points(quad_pts)
        # Quantities to interpolate
        G = field.G()
        iota = field.iota()
        modB = field.modB()
        modB_derivs = field.modB_derivs()
        quad_info = np.hstack((modB, modB_derivs, G, iota))
        quad_info = np.ascontiguousarray(quad_info)
        psi0 =field.psi0

        # print(stz)
        # print("calculating new interpolation")
        # interpolated_values = sopp.test_interpolation(quad_info, srange, trange, zrange, stz, 6)
        # print(stz)
        # print(interpolated_values)

        print("calculating new derivatives")
        # new_derivs = np.empty((n_test_pts, 4))
        # for i in range(n_test_pts):
        #         new_derivs[i,:] = sopp.test_derivatives(quad_info, srange, trange, zrange, stz[i,:], vpar_init[i], VELOCITY, MASS, CHARGE, psi0)

        new_derivs = sopp.test_derivatives(quad_info, srange, trange, zrange, stz, vpar_init, VELOCITY, MASS, CHARGE, psi0, stz.shape[0])
        new_derivs = np.reshape(new_derivs, (stz.shape[0], 4))

        print("simsopt derivatives: ", old_derivs)
        print("new derivatives: ", new_derivs)
        rel_err = np.abs((old_derivs - new_derivs) / old_derivs)
        diff = np.max(rel_err)
        print(np.abs(old_derivs - new_derivs) / old_derivs)
        print("diff=", diff)
        # print(stz)
        print("Maximum relative error in derivative values on {} points: {}".format(n_test_pts, diff))

        print("culprit particle:")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(stz[row_index, :])
        print(vpar_init[row_index])
        print("simsopt", old_derivs[row_index, :])
        print("new", new_derivs[row_index, :])
        print(rel_err[row_index, :])





def test_timestep():
        n_metagrid_pts = 15


        # create a B-field
        filename = os.path.join('/global/homes/m/mczek/simsopt/examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')
        vmec = Vmec(filename)

        order = 3
        bri = BoozerRadialInterpolant(vmec, order, enforce_vacuum=True)

        nfp = vmec.wout.nfp
        degree = 3
        srange = (0, 1, n_metagrid_pts)
        thetarange = (0, np.pi, n_metagrid_pts)
        zetarange = (0, 2*np.pi/nfp, n_metagrid_pts)
        field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

        # generate test points
        n_test_pts = 10000
        s = np.random.uniform(low=0, high=0.99, size=(n_test_pts,1))
        t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        stz = np.hstack((s,t,z))

        # stz = np.array([[0.99822594, 3.43868632, 5.04501761]])
        # print("stz")
        # print(stz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        # vpar_init = np.array([-8932370.0722737])

        print("computing simsopt timestep")

        
        gc_tys, gc_zeta_hits = trace_particles_boozer(
                field, stz, vpar_init, tmax=1e-2, mass=MASS, charge=CHARGE,
                Ekin=ENERGY, zetas=[0], tol=1e-9, stopping_criteria=[IterationStoppingCriterion(1)],
                forget_exact_path=True)
        
        print(gc_tys)
        final_positions = np.array([x[-1] for x in gc_tys])
        # print(np.array(final_positions))

        print("computing new timesteps")
        srange = (0, 1, 3*n_metagrid_pts+1)
        thetarange = (0, np.pi, 3*n_metagrid_pts+1)
        zetarange = (0, 2*np.pi/nfp, 3*n_metagrid_pts+1)

        s_grid = np.linspace(srange[0], srange[1], srange[2])
        theta_grid = np.linspace(thetarange[0], thetarange[1], thetarange[2])
        zeta_grid = np.linspace(zetarange[0], zetarange[1], zetarange[2])

        # print("theta_grid", theta_grid)
        # print("zeta_grid", zeta_grid)


        print("building quad_pts")
        quad_pts = np.empty((srange[2]*thetarange[2]*zetarange[2], 3))
        for i in range(srange[2]):
                for j in range(thetarange[2]):
                        for k in range(zetarange[2]):
                                quad_pts[thetarange[2]*zetarange[2]*i + zetarange[2]*j + k, :] = [s_grid[i], theta_grid[j], zeta_grid[k]]


        print("building interpolation info")
        field.set_points(quad_pts)
        G = field.G()
        iota = field.iota()
        I = field.I()
        modB = field.modB()
        J = (G + iota*I)/(modB**2)
        # minJ = np.min(J)
        maxJ = np.max(J)
        # print("maxJ", maxJ)

        psi0 = field.psi0
        modB_derivs = field.modB_derivs()

        quad_info = np.hstack((modB, modB_derivs, G, iota))
        quad_info = np.ascontiguousarray(quad_info)

        last_time = sopp.test_timestep(
                quad_pts=quad_info, 
                srange=srange,
                trange=thetarange,
                zrange=zetarange, 
                stz_init=stz,
                m=MASS, 
                q=CHARGE, 
                vtotal=np.sqrt(2*ENERGY/MASS),  
                vtang=vpar_init, 
                tol=1e-9, 
                psi0=psi0, 
                nparticles=n_test_pts)


        last_time = np.reshape(last_time, (n_test_pts, 7))
        # print(last_time)

        new_final_positions = np.array([[x[4], x[0], x[1], x[2], x[3]] for x in last_time])
        # print(np.array(new_final_positions))

        print("simsopt final position: ", final_positions)
        print("new final position: ", new_final_positions)
        rel_err = np.abs((final_positions - new_final_positions) / final_positions)
        diff = np.max(rel_err)
        print(np.abs(final_positions - new_final_positions) / final_positions)
        print("diff=", diff)
        # print(stz)
        print("Maximum relative error in final positions on {} points: {}".format(n_test_pts, diff))

        print("culprit particle:")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(stz[row_index, :])
        print(vpar_init[row_index])
        print("simsopt", final_positions[row_index, :])
        print("new", new_final_positions[row_index, :])
        print(rel_err[row_index, :])

# test_derivs(15)

test_timestep()