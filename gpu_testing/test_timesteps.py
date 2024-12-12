import simsoptpp as sopp
from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec
import numpy as np
from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )
import os

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
        field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

        # generate test points
        n_test_pts = 10000
        s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
        t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
        stz = np.hstack((s,t,z))

        print(stz)

        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))

        print("computing simsopt interpolant")

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
        new_derivs = np.empty((n_test_pts, 4))
        for i in range(n_test_pts):
                new_derivs[i,:] = sopp.test_derivatives(quad_info, srange, trange, zrange, stz[i,:], vpar_init[i], VELOCITY, MASS, CHARGE, psi0)


        print("simsopt derivatives: ", old_derivs)
        print("new derivatives: ", new_derivs)
        diff = np.max(np.abs(old_derivs - new_derivs) / old_derivs)
        print("diff=", diff)
        # print(stz)
        print("Maximum difference in derivatives values on {} points: {}".format(n_test_pts, diff))



test_derivs(15)