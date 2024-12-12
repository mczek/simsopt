import numpy as np
import unittest
import simsoptpp as sopp
from numpy.testing import assert_raises
import numpy as np
import os


from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)
from simsopt.mhd import Vmec

def get_random_polynomial(dim, degree):
    coeffsx = np.random.standard_normal(size=(degree+1, dim))
    coeffsy = np.random.standard_normal(size=(degree+1, dim))
    coeffsz = np.random.standard_normal(size=(degree+1, dim))

    def fun(x, y, z, flatten=True):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        px = sum([coeffsx[i, :] * x[:, None]**i for i in range(degree+1)])
        py = sum([coeffsy[i, :] * y[:, None]**i for i in range(degree+1)])
        pz = sum([coeffsz[i, :] * z[:, None]**i for i in range(degree+1)])
        res = px*py*pz
        if flatten:
            return (np.ascontiguousarray(res)).flatten()
        else:
            return res
    return fun


def shape(x):
    shape = [0,0,0,0]
    shape[0] = (1.0-x)*(2.0-x)*(3.0-x)/6.0;
    shape[1] = x*(2.0-x)*(3.0-x)/2.0;
    shape[2] = x*(x-1.0)*(3.0-x)/2.0;
    shape[3] = x*(x-1.0)*(x-2.0)/6.0;
    return shape

def my_interpolant(xyz, xran, yran, zran, fun):


    print("test")
    xran = (xran[0],  xran[1], xran[2]+1)
    yran = (yran[0],  yran[1], yran[2]+1)
    zran = (zran[0],  zran[1], zran[2]+1)


    x_grid = np.linspace(xran[0], xran[1], xran[2])
    # print(xran)
    # print(x_grid)
    y_grid = np.linspace(yran[0], yran[1], yran[2])
    z_grid = np.linspace(zran[0], zran[1], zran[2])

    quad_pts = np.empty((xran[2]*yran[2]*zran[2], 3))
    for i in range(xran[2]):
        for j in range(yran[2]):
            for k in range(zran[2]):
                quad_pts[yran[2]*zran[2]*i + zran[2]*j + k] = [x_grid[i], y_grid[j], z_grid[k]]

    quad_info = fun(quad_pts[:,0], quad_pts[:, 1], quad_pts[:,2], flatten=False)
    output = np.zeros(xyz.shape)

    for i in range(xyz.shape[0]):
        loc = xyz[i,:]
        interpolated_values = sopp.test_interpolation(quad_info, xran, yran, zran, loc, 3)
        # print(interpolated_values)
        output[i,:] = interpolated_values
    return output

def subtest_regular_grid_interpolant_exact(dim, degree):
        """
        Build a random, vector valued polynomial of a specific degree and check
        that it is interpolated exactly.
        """
        np.random.seed(1800)
        xran = (1.0, 4.0, 10)
        yran = (1.1, 3.9, 10)
        zran = (1.2, 3.8, 10)

        fun = get_random_polynomial(dim, degree)

        rule = sopp.UniformInterpolationRule(degree)

        interpolant = sopp.RegularGridInterpolant3D(rule, xran, yran, zran, dim, True)
        interpolant.interpolate_batch(fun)

        nsamples = 10000
        xpoints = np.random.uniform(low=xran[0], high=xran[1], size=(nsamples, ))
        ypoints = np.random.uniform(low=yran[0], high=yran[1], size=(nsamples, ))
        zpoints = np.random.uniform(low=zran[0], high=zran[1], size=(nsamples, ))
        xyz = np.asarray([xpoints, ypoints, zpoints]).T.copy()

        # print("xyz", xyz)

        fhxyz = np.zeros((nsamples, dim))
        fxyz = fun(xyz[:, 0], xyz[:, 1], xyz[:, 2], flatten=False)

        interpolant.evaluate_batch(xyz, fhxyz)

        xran = (1.0, 4.0, 30)
        yran = (1.1, 3.9, 30)
        zran = (1.2, 3.8, 30)
        fhxyz_mine = my_interpolant(xyz, xran, yran, zran, fun)

        # print("error", fxyz - fhxyz_mine)
        # assert np.allclose(fxyz, fhxyz_mine, atol=1e-12, rtol=1e-12)
        print("Polynomial interpolation difference on {} points: {}".format(nsamples, np.max(np.abs((fhxyz-fhxyz_mine)))))
        # print(np.max(np.abs((fxyz-fhxyz_mine))))

        # assert np.allclose(fxyz, fhxyz, atol=1e-12, rtol=1e-12)
        # print(np.max(np.abs((fxyz-fhxyz))))

        # print()


def test_interpolant_bfield(n_metagrid_pts):
    
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
    np.random.seed(1800)

    s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
    t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    stz = np.hstack((s,t,z))


    # SIMSOPT INTERPOLANT
    print(stz)
    field.set_points(stz)
    G = field.G()
    iota = field.iota()
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    simsopt_interpolation = np.hstack((modB, modB_derivs, G, iota))


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

    # Calculate interpolation
    new_interpolation = np.zeros((stz.shape[0], 6))
    for i in range(stz.shape[0]):
        loc = stz[i,:]
        # while loc[2] > 2*np.pi/nfp:
        #     loc[2] -= 2*np.pi/nfp
        # symm_exploited = False
        # if loc[1] > np.pi:
        #     period = 2*np.pi / nfp
        #     loc[2] = period - loc[2]
        #     loc[1] = 2*np.pi - loc[1]
        interpolated_values = sopp.test_interpolation(quad_info, srange, trange, zrange, loc, 6)

        # if symm_exploited:
        #     interpolated_values[2] *= -1
        #     interpolated_values[3] *= -1
        # print(interpolated_values)
        new_interpolation[i,:] = interpolated_values

    print(np.abs(simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    diff =np.max(np.abs(simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    print("Maximum difference in interpolation values on {} points: {}".format(n_test_pts, diff))



# subtest_regular_grid_interpolant_exact(3, 3)
test_interpolant_bfield(15)