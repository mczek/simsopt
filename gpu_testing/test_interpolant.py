import numpy as np
import unittest
import simsoptpp as sopp
from numpy.testing import assert_raises
import numpy as np
import os
from simsopt._core import load
from simsopt.geo import SurfaceXYZTensorFourier

from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec

def setup_interpolant(field, nfp, n_metagrid_pts):
    ### NEW INTERPOLANT

    srange = (0, 1.0, 3*n_metagrid_pts+1)
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

    return srange, trange, zrange, quad_info


def test_interpolant_bfield(field, nfp, n_metagrid_pts, n_test_pts, verify=True):

    # generate test points
    s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
    t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    stz = np.hstack((s,t,z))



    ## NEW INTERPOLANT
    srange, trange, zrange, quad_info = setup_interpolant(field, nfp, n_metagrid_pts)
    stz = np.ascontiguousarray(stz)

    # Calculate interpolation

    new_interpolation = sopp.test_gpu_interpolation(quad_info, srange, trange, zrange, stz, 6, stz.shape[0])
    new_interpolation = np.reshape(new_interpolation, (stz.shape[0], 6))

    if verify:
        # SIMSOPT INTERPOLANT
        field.set_points(stz)
        G = field.G()
        iota = field.iota()
        modB = field.modB()
        modB_derivs = field.modB_derivs()
        simsopt_interpolation = np.hstack((modB, modB_derivs, G, iota))

        print(np.abs(simsopt_interpolation - new_interpolation) / simsopt_interpolation)
        rel_err = np.abs((simsopt_interpolation - new_interpolation) / simsopt_interpolation)
        diff = np.max(rel_err)
        print("Maximum relative error in interpolation values on {} points: {}".format(n_test_pts, diff))

        print("culprit particle:")
        row_index = np.argmax(rel_err) // rel_err.shape[1]
        print(stz[row_index, :])
        print("simsopt", simsopt_interpolation[row_index, :])
        print("new", new_interpolation[row_index, :])
        print(rel_err[row_index, :])


if __name__ == "__main__":
    ### CREATE A FIELD FOR TRACING
    filename = os.path.join('./examples/2_Intermediate/inputs/input.LandremanPaul2021_QH')
    vmec = Vmec(filename)

    order = 3
    bri = BoozerRadialInterpolant(vmec, order, enforce_vacuum=True)
    nfp = vmec.wout.nfp
    degree = 3
    n_metagrid_pts = 15
    srange = (0, 1, n_metagrid_pts)
    thetarange = (0, np.pi, n_metagrid_pts)
    zetarange = (0, 2*np.pi/nfp, n_metagrid_pts)
    field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

    test_interpolant_bfield(field, nfp, n_metagrid_pts, 100000, True)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 200000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 400000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 800000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 1600000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 3200000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 6400000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 12800000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 25600000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 51200000, False)
    test_interpolant_bfield(field, nfp, n_metagrid_pts, 102400000, False)
    # Maximum relative error in interpolation values on 100000 points: 5.162441928301709e-09, time = 1.2857ms
    # 200,000 pts:      0.61456ms
    # 400,000 pts:      1.0288ms
    # 800,000 pts:      1.93123ms
    # 1,600,000 pts:    3.74173ms
    # 3,200,000 pts:    7.48771ms
    # 6,400,000 pts:    14.6771ms
    # 12,800,000 pts:   29.198ms
    # 25,600,000 pts:   57.9434ms
    # 51,200,000 pts:   109.39ms
    # 102,400,000 pts:  203.205ms

    # Remove in-bounds checks from interpolant (~10% speedup)
    # Maximum relative error in interpolation values on 100000 points: 1.3650731041219012e-08, time = 1.7369ms
    # 200,000 pts:      0.519072ms
    # 400,000 pts:      0.915936ms
    # 800,000 pts:      1.73315ms
    # 1,600,000 pts:    3.28528ms
    # 3,200,000 pts:    6.64227ms
    # 6,400,000 pts:    12.8269ms
    # 12,800,000 pts:   25.2023ms
    # 25,600,000 pts:   50.0086ms
    # 51,200,000 pts:   99.6863ms
    # 102,400,000 pts:  183.34ms

    # Remove control flow on p.symmetry_exploited (no speedup, maybe slow down)
    # Note that this doesn't change the interpolate function itself, but is used in the testing kernel
    # Maximum relative error in interpolation values on 100000 points: 1.965625717004303e-09, time = 1.67802ms
    # 200,000 pts:      0.525536
    # 400,000 pts:      0.908352
    # 800,000 pts:      1.7512
    # 1,600,000 pts:    3.29411ms
    # 3,200,000 pts:    6.84218ms
    # 6,400,000 pts:    12.8062ms
    # 12,800,000 pts:   25.1273ms
    # 25,600,000 pts:   49.7635ms
    # 51,200,000 pts:   99.2159ms
    # 102,400,000 pts:  191.744ms


    # Remove redundant arithmetic in the interpolant (small speed up for largest input)
    # Maximum relative error in interpolation values on 100000 points: 1.5997891471777334e-08, time = 1.71331ms
    # 200,000 pts:      0.520384
    # 400,000 pts:      0.901888
    # 800,000 pts:      1.76448
    # 1,600,000 pts:    3.24842ms
    # 3,200,000 pts:    6.71773ms
    # 6,400,000 pts:    12.7971ms
    # 12,800,000 pts:   25.088ms
    # 25,600,000 pts:   49.8016ms
    # 51,200,000 pts:   99.2916ms
    # 102,400,000 pts:  179.883ms