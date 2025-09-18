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
import os
from test_interpolant_cartesian import *

np.random.seed(1800)


def test_derivs(field, sc_praticle, nfp, n_metagrid_pts, n_test_pts, verify=True):
        # generate test points
        r_range = field.r_range
        phi_range = field.phi_range
        z_range = field.z_range

        # generate test points
        r = np.random.uniform(low=r_range[0]+0.1, high=r_range[1]-0.1, size=(n_test_pts,1))
        phi = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=-z_range[1]+0.1, high=z_range[1]-0.1, size=(n_test_pts,1))
        rphiz = np.hstack((r,phi,z))
        # rphiz = np.array([[1.64982937,  4.57359021, -0.05896898]])
        rphiz = np.ascontiguousarray(rphiz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        # vpar_init = np.array([4355542.313737903])

      

        ### NEW INTERPOLANT
        r_range, phi_range, z_range, quad_info = setup_interpolant(field, sc_particle, nfp, n_metagrid_pts)

        # psi0 =field.psi0

        print("calculating new derivatives")
        new_derivs = sopp.test_derivatives_cartesian(quad_info, r_range, phi_range, z_range, rphiz, vpar_init, VELOCITY, MASS, CHARGE, rphiz.shape[0])
        new_derivs = np.reshape(new_derivs, (rphiz.shape[0], 4))

        if verify:
                print("computing simsopt derivatives")
                old_derivs = np.empty((n_test_pts, 4))
                for i in range(n_test_pts):
                        old_derivs[i,:] = sopp.simsopt_derivs_cartesian(field, rphiz[i,:], MASS, CHARGE, VELOCITY, vpar_init[i])

                dist_fn = sc_particle.evaluate_rphiz(rphiz)[:, 0]
                print(dist_fn)
                rel_err = np.abs((old_derivs - new_derivs) / old_derivs)
                rel_err = rel_err[dist_fn > 0, :] # only consider particles inside the device
                diff = np.max(rel_err)
                print(np.abs(old_derivs - new_derivs) / old_derivs)

                print("Maximum relative error in derivative values on {} points: {}".format(rel_err.shape[0], diff))

                print("culprit particle:")
                row_index = np.argmax(rel_err) // rel_err.shape[1]
                print(rphiz[row_index, :])
                print(vpar_init[row_index])
                print("simsopt", old_derivs[row_index, :])
                print("new", new_derivs[row_index, :])
                print(rel_err[row_index, :])





def test_timestep(field, sc_particle, nfp, n_metagrid_pts, n_test_pts, verify=True):

        # generate test points
        r_range = field.r_range
        phi_range = field.phi_range
        z_range = field.z_range

        # generate test points
        r = np.random.uniform(low=r_range[0]+0.1, high=r_range[1]-0.1, size=(n_test_pts,1))
        phi = np.random.uniform(low=phi_range[0], high=phi_range[1], size=(n_test_pts,1))
        z = np.random.uniform(low=z_range[0]+0.1, high=z_range[1]-0.1, size=(n_test_pts,1))
        rphiz = np.hstack((r,phi,z))
        rphiz = np.ascontiguousarray(rphiz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        vpar_init = np.ascontiguousarray(vpar_init)



        ### NEW INTERPOLANT
        print("setting up new interpolant")
        r_range, phi_range, z_range, quad_info = setup_interpolant(field, sc_particle, nfp, n_metagrid_pts)
        quad_info = np.ascontiguousarray(quad_info)
        # # for i in range(n_test_pts):
        # print(r_range)
        # print(phi_range),
        # print(z_range)
        # rphiz_test = rphiz[i, :]
        # vpar_init_test = vpar_init[i]

        # rphiz = np.array([[1.36825919, 0.61279615, 0.26253024]])
        # vpar_init = [-6203622.275269774]
        print("rphiz", rphiz)

        # print(i, rphiz_test, vpar_init_test)
        # print(quad_info.shape)
        print("testing new timstep")
        
        last_time = sopp.test_timestep_cartesian(
                quad_pts=quad_info, 
                srange=r_range,
                trange=phi_range,
                zrange=z_range, 
                stz_init=rphiz,
                m=MASS, 
                q=CHARGE, 
                vtotal=np.sqrt(2*ENERGY/MASS),  
                vtang=vpar_init, 
                tol=1e-9, 
                nparticles=n_test_pts)


        # print("last_time", last_time)

        last_time = np.reshape(last_time, (n_test_pts, 7))

        # print("last_time", last_time)
        # print(last_time[:, 4])

        new_final_positions = np.array([[x[4], x[0], x[1], x[2], x[3]] for x in last_time])

        # print("new final positions", new_final_positions)

        print("computing simsopt timestep")

        if verify:
                r = rphiz[:, 0].reshape(-1,1)
                phi = rphiz[:, 1].reshape(-1,1)
                z = rphiz[:, 2].reshape(-1,1)
                x = r*np.cos(phi)
                y = r*np.sin(phi)
                print(x,y,z)
                xyz = np.hstack((x,y,z))
                print(xyz)
                print(vpar_init)
                print(xyz.shape)
                print(len(vpar_init))
                gc_tys, gc_zeta_hits = trace_particles(
                        field, xyz, vpar_init, tmax=1e-2, mass=MASS, charge=CHARGE,
                        Ekin=ENERGY, tol=1e-9, stopping_criteria=[IterationStoppingCriterion(1)],
                        forget_exact_path=True)
                print("done with simsopt timestep")
                # print(gc_tys)
                # print(gc_zeta_hits)
                final_positions = np.array([x[-1] for x in gc_tys])
                rel_err = np.abs((final_positions - new_final_positions) / final_positions)
                diff = np.max(rel_err)
                # print(np.abs(final_positions - new_final_positions) / final_positions)

                print("Maximum relative error in final positions on {} points: {}".format(n_test_pts, diff))

                print("culprit particle:")
                row_index = np.argmax(rel_err) // rel_err.shape[1]
                print(rphiz[row_index, :])
                print(vpar_init[row_index])
                print("simsopt", final_positions[row_index, :])
                print("new", new_final_positions[row_index, :])
                print(rel_err[row_index, :])


if __name__ == "__main__":
        ### CREATE A FIELD FOR TRACING
        nfp = 3
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, nfp, True)
        curves = [c.curve for c in coils]
        bs = BiotSavart(coils)
        # proc0_print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
        # proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

        mpol = 5
        ntor = 5
        stellsym = True
        s = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                                range="full torus", nphi=64, ntheta=24)
        s.fit_to_curve(ma, 0.20, flip_theta=False)

        n_metagrid_pts = 30
        degree = 3
        rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
        zs = s.gamma()[:, :, 2]
        sc_particle = SurfaceClassifier(s, h=0.1, p=2)


        rrange = (np.min(rs), np.max(rs), n_metagrid_pts)
        phirange = (0, 2*np.pi/nfp, n_metagrid_pts)
        # exploit stellarator symmetry and only consider positive z values:
        zrange = (0, np.max(zs), n_metagrid_pts)
        print(rrange, phirange, zrange)
        bsh = InterpolatedField(
                bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
        )
        np.random.seed(1800)
        # test_interpolant_bfield(bsh, sc_particle, nfp, n_metagrid_pts, 100000)

        test_derivs(bsh, sc_particle, nfp, n_metagrid_pts, 100000)

        test_timestep(bsh, sc_particle, nfp, n_metagrid_pts, 100000)


        # derivs baseline
        # calculating new derivatives
        # interpolation kernel time (ms): 0.89344
        # calculating new derivatives
        # interpolation kernel time (ms): 1.72909
        # calculating new derivatives
        # interpolation kernel time (ms): 3.24525
        # calculating new derivatives
        # interpolation kernel time (ms): 6.25683
        # calculating new derivatives
        # interpolation kernel time (ms): 12.4094


        # single timestep baseline
        # tracing kernels time (ms): 10.0615
        # tracing kernels time (ms): 19.1095
        # tracing kernels time (ms): 36.6104
        # tracing kernels time (ms): 72.267
        # tracing kernels time (ms): 143.349
        # tracing kernels time (ms): 285.381

        # starting particle tracing kernel (with local particle)
        # tracing kernels time (ms): 16.955
        # tracing kernels time (ms): 33.0403
        # tracing kernels time (ms): 65.5089
        # tracing kernels time (ms): 130.325
        # tracing kernels time (ms): 259.851