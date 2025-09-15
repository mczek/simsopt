import numpy as np
import unittest
import simsoptpp as sopp
from numpy.testing import assert_raises
import numpy as np
import os
from simsopt._core import load
from simsopt.geo import SurfaceRZFourier
from simsopt.configs import get_ncsx_data
from simsopt.field import (BiotSavart, InterpolatedField, coils_via_symmetries, trace_particles_starting_on_curve,
                           SurfaceClassifier, LevelsetStoppingCriterion, plot_poincare_data)

def setup_interpolant(field, sc_particle, nfp, n_metagrid_pts):
    ### NEW INTERPOLANT

    # srange = (0, 1.0, 3*n_metagrid_pts+1)
    # trange = (0, np.pi, 3*n_metagrid_pts+1)
    # zrange = (0, 2*np.pi/nfp, 3*n_metagrid_pts+1)
    r_range = (field.r_range[0], field.r_range[1], 3*field.r_range[2]+1)
    phi_range = (field.phi_range[0], field.phi_range[1], 3*field.phi_range[2]+1)
    z_range = (field.z_range[0], field.z_range[1], 3*field.z_range[2]+1)

    r_grid = np.linspace(r_range[0], r_range[1], r_range[2])
    phi_grid = np.linspace(phi_range[0], phi_range[1], phi_range[2])
    z_grid = np.linspace(z_range[0], z_range[1], z_range[2])

    quad_pts = np.empty((r_range[2]*phi_range[2]*z_range[2], 3))
    for i in range(r_range[2]):
        for j in range(phi_range[2]):
            for k in range(z_range[2]):
                quad_pts[phi_range[2]*z_range[2]*i + z_range[2]*j + k, :] = [r_grid[i], phi_grid[j], z_grid[k]]



    field.set_points_cyl(quad_pts)

    # Quantities to interpolate
    B = field.B_cyl()
    GradAbsB = field.GradAbsB_cyl()
        
    signed_dist_vals = sc_particle.evaluate_rphiz(quad_pts)

    quad_info = np.hstack((B, GradAbsB, signed_dist_vals))

    # reorder for device memory accesses
    print("reordering interpolant data form device accesses")
    cell_quad_pts = np.empty((field.r_range[2]*field.z_range[2]*field.phi_range[2]*64, quad_info.shape[1]))
    for cell_r in range(field.r_range[2]):
        for cell_phi in range(field.phi_range[2]):
            for cell_z in range(field.z_range[2]):
                row_start = 64*(cell_r*field.phi_range[2]*field.z_range[2] + cell_phi*field.z_range[2] + cell_z)

                # if cell_r == 24 and cell_phi == 22 and cell_z == 20:
                #     print(row_start)

                assert 3*cell_r + i < r_range[2]
                # iterate over spline locations for this cell
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            row_idx = row_start + 16*i + 4*j + k

                            # if cell_r == 24 and cell_phi == 22 and cell_z == 20:
                            #     print(row_idx)
                            # print(3*cell_r+i, 3*cell_phi+j, 3*cell_z+k)
                            # print(field.r_range[2], field.phi_range[2], field.z_range[2])
                            # print(row_idx, phi_range[2]*z_range[2]*(3*cell_r + i) + z_range[2]*(3*cell_phi+j) + 3*cell_z + k)
                            cell_quad_pts[row_idx,: ] = quad_info[phi_range[2]*z_range[2]*(3*cell_r + i) + z_range[2]*(3*cell_phi+j) + 3*cell_z + k, :]

    cell_quad_pts = np.ascontiguousarray(cell_quad_pts)

    return r_range, phi_range, z_range, cell_quad_pts


def test_interpolant_bfield(field, sc_particle, nfp, n_metagrid_pts, n_test_pts):

    r_range = field.r_range
    phi_range = field.phi_range
    z_range = field.z_range

    # generate test points
    r = np.random.uniform(low=r_range[0], high=r_range[1], size=(n_test_pts,1))
    phi = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    z = np.random.uniform(low=-z_range[1], high=z_range[1], size=(n_test_pts,1))
    rphiz = np.hstack((r,phi,z))
    # rphiz = np.array([[ 1.16373528,  0.04877549, -0.13471403]])
    rphiz = np.ascontiguousarray(rphiz)

    # SIMSOPT INTERPOLANT
    field.set_points_cyl(rphiz)
    B = field.B_cyl()
    GradAbsB = field.GradAbsB_cyl()
    signed_dist_vals = sc_particle.evaluate_rphiz(rphiz)
    simsopt_interpolation = np.hstack((B, GradAbsB, signed_dist_vals))

    ## NEW INTERPOLANT
    r_range, phi_range, z_range, quad_info = setup_interpolant(field, sc_particle, nfp, n_metagrid_pts)

    # Calculate interpolation
    new_interpolation = sopp.test_gpu_interpolation(quad_info, r_range, phi_range, z_range, rphiz, "cartesian", rphiz.shape[0])
    new_interpolation = np.reshape(new_interpolation, (rphiz.shape[0], 7))

    print(np.abs(simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    rel_err = np.abs((simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    dist_fn = simsopt_interpolation[:, 6]
    rel_err = rel_err[dist_fn > 0, :-1] # don't test boundary distance for now
    diff = np.max(rel_err)
    print("Maximum relative error in interpolation values on {} points: {}".format(rel_err.shape[0], diff))

    print("culprit particle:")
    row_index = np.argmax(rel_err) // rel_err.shape[1]
    print(rphiz[row_index, :])
    print("simsopt", simsopt_interpolation[row_index, :])
    print("new", new_interpolation[row_index, :])
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
    test_interpolant_bfield(bsh, sc_particle, nfp, n_metagrid_pts, 100000)