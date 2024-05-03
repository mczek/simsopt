
import time
import os
import logging
import sys
import numpy as np
from math import sqrt

import simsoptpp as sopp


from simsopt.configs import get_ncsx_data
from simsopt.field import (BiotSavart, InterpolatedField, coils_via_symmetries, trace_particles_starting_on_curve,
                           SurfaceClassifier, LevelsetStoppingCriterion, plot_poincare_data)
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import in_github_actions, proc0_print, comm_world
from simsopt._core.util import parallel_loop_bounds
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV, ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from simsopt.field.sampling import draw_uniform_on_curve

proc0_print("Running 1_Simple/tracing_particle.py")
proc0_print("====================================")

sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# If we're in the CI, make the run a bit cheaper:
nparticles = 3 if in_github_actions else 1
degree = 2 if in_github_actions else 3

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

nfp = 3
curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
proc0_print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))
curves_to_vtk(curves + [ma], OUT_DIR + 'coils')

mpol = 5
ntor = 5
stellsym = False
s = SurfaceRZFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                      range="full torus", nphi=64, ntheta=24)


s.fit_to_curve(ma, 0.20, flip_theta=False)
sc_particle = SurfaceClassifier(s, h=0.1, p=2)
n = 16
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]

print("rs", rs)
print("zs", zs)

rrange = (np.min(rs), np.max(rs), 2*n)
phirange = (0, 2*np.pi, n*2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (np.min(zs), np.max(zs), 2*n)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
)

### tracing functions to use gpu

def trace_particles_gpu(field,
                    xyz_inits,
                    parallel_speeds,  
                    tmax=1e-4,
                    mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                    tol=1e-9, comm=None, phis=[], stopping_criteria=[], mode='gc_vac', forget_exact_path=False,
                    phase_angle=0):
    r"""
    Follow particles in a magnetic field.

    In the case of ``mod='full'`` we solve

    .. math::

        [\ddot x, \ddot y, \ddot z] = \frac{q}{m}  [\dot x, \dot y, \dot z] \times B

    in the case of ``mod='gc_vac'`` we solve the guiding center equations under
    the assumption :math:`\nabla p=0`, that is

    .. math::

        [\dot x, \dot y, \dot z] &= v_{||}\frac{B}{|B|} + \frac{m}{q|B|^3}  (0.5v_\perp^2 + v_{||}^2)  B\times \nabla(|B|)\\
        \dot v_{||}    &= -\mu  (B \cdot \nabla(|B|))

    where :math:`v_\perp = 2\mu|B|`. See equations (12) and (13) of
    [Guiding Center Motion, H.J. de Blank, https://doi.org/10.13182/FST04-A468].

    Args:
        field: The magnetic field :math:`B`.
        xyz_inits: A (nparticles, 3) array with the initial positions of the particles.
        parallel_speeds: A (nparticles, ) array containing the speed in direction of the B field
                         for each particle.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, M)
            with M depending on the ``mode``.  Each row contains the time and
            the state.  So for `mode='gc'` and `mode='gc_vac'` the state
            consists of the xyz position and the parallel speed, hence
            each row contains `[t, x, y, z, v_par]`.  For `mode='full'`, the
            state consists of position and velocity vector, i.e. each row
            contains `[t, x, y, z, vx, vy, vz]`.

        - ``res_phi_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the phi planes or
            one of the stopping criteria. Each row of the array contains
            `[time] + [idx] + state`, where `idx` tells us which of the `phis`
            or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
    """

    nparticles =  xyz_inits.shape[0]
    assert xyz_inits.shape[0] == len(parallel_speeds)
    speed_par = parallel_speeds
    mode = mode.lower()
    assert mode in ['gc', 'gc_vac', 'full']
    m = mass
    speed_total = sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)

    # if mode == 'full':
    #     xyz_inits, v_inits, _ = gc_to_fullorbit_initial_guesses(field, xyz_inits, speed_par, speed_total, m, charge, eta=phase_angle)
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0

    # parallelization
    print(type(xyz_inits))
    print(xyz_inits)
    print(speed_par)

    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi, n*2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (np.min(zs), np.max(zs), n//2)

    r_vals = np.linspace(rrange[0], rrange[1], rrange[2])
    phi_vals = np.linspace(phirange[0], phirange[1], phirange[2])
    z_vals = np.linspace(zrange[0], zrange[1], zrange[2])

    quad_pts = np.empty((rrange[2]*phirange[2]*zrange[2], 3))
    for i in range(rrange[2]):
        for j in range(zrange[2]):
            for k in range(phirange[2]):
                quad_pts[zrange[2]*phirange[2]*i + phirange[2]*j + k, :] = [r_vals[i], z_vals[j], phi_vals[k]]

    print(rrange[2]*zrange[2]*phirange[2])
    print(quad_pts.shape)
    print(quad_pts)
    print(rrange)
    print("r", r_vals)
    print("z", z_vals)
    print("phi", phi_vals)


    # exit()
    # rr, zz, phiphi= np.meshgrid(r_vals, z_vals, phi_vals)
    quad_pts = np.ascontiguousarray(quad_pts)
    nquadpts = quad_pts.shape[0]

    # B interpolation pts
    bsh.set_points_cyl(quad_pts)
    quad_B = bsh.B() # note this is in cartesian coords

    # print(quad_B)
    # Surface interpolation using same quad pts
    rphiz_quadpts = np.ascontiguousarray(np.array((quad_pts[:, 0], quad_pts[:, 2], quad_pts[:, 1])).T)
    print(rphiz_quadpts)
    vals = sc_particle.evaluate_rphiz(rphiz_quadpts)
    print("output vals")
    print(vals)
    vals = vals.reshape((quad_pts[:,0].shape[0], 1))
    print("vals")
    print(vals)
    print("max vals", max(vals))
    print(vals[2573])

    # exit()

    init_dists = sc_particle.evaluate_xyz(xyz_inits)
    print(init_dists)
    # exit()

    quad_info = np.hstack((quad_B, vals))
    print(quad_B.shape)
    print(vals.shape)
    print("quad info")
    print(quad_info)
    print(quad_info[2573, :])

    final_pos = sopp.gpu_tracing(
        quad_info, rrange,  phirange, zrange, xyz_inits,
        m, charge, speed_total, speed_par, tmax, tol,
        vacuum=(mode == 'gc_vac'), phis=phis, stopping_criteria=stopping_criteria, nparticles=nparticles)

    final_dist = sc_particle.evaluate_xyz(np.reshape(final_pos, (nparticles,3)))
    did_leave = [x < 0 for x in final_dist]
    print("printing output")
    loss_ctr = sum(did_leave)
    logger.debug(f'Particles lost {loss_ctr}/{nparticles}={(100*loss_ctr)//nparticles:d}%')
    return did_leave

def trace_particles_starting_on_curve_gpu(curve, field, nparticles, tmax=1e-4,
                                      mass=ALPHA_PARTICLE_MASS, charge=ALPHA_PARTICLE_CHARGE,
                                      Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
                                      tol=1e-9, comm=None, seed=1, umin=-1, umax=+1,
                                      phis=[], stopping_criteria=[], mode='gc_vac', forget_exact_path=False,
                                      phase_angle=0):
    r"""
    Follows particles spawned at random locations on the magnetic axis with random pitch angle.
    See :mod:`simsopt.field.tracing.trace_particles` for the governing equations.

    Args:
        curve: The :mod:`simsopt.geo.curve.Curve` to spawn the particles on. Uses rejection sampling
               to sample points on the curve. *Warning*: assumes that the underlying
               quadrature points on the Curve are uniformly distributed.
        field: The magnetic field :math:`B`.
        nparticles: number of particles to follow.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        seed: random seed
        umin: the parallel speed is defined as  ``v_par = u * speed_total``
              where  ``u`` is drawn uniformly in ``[umin, umax]``
        umax: see ``umin``
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: see :mod:`simsopt.field.tracing.trace_particles`
    """
    m = mass
    speed_total = sqrt(2*Ekin/m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    np.random.seed(seed)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles, ))
    speed_par = us*speed_total
    xyz, _ = draw_uniform_on_curve(curve, nparticles, safetyfactor=10)
    return trace_particles_gpu(
        field, xyz, speed_par, tmax=tmax, mass=mass, charge=charge,
        Ekin=Ekin, tol=tol, comm=comm, phis=phis,
        stopping_criteria=stopping_criteria, mode=mode, forget_exact_path=forget_exact_path,
        phase_angle=phase_angle)



def trace_particles(bfield, label, mode='gc_vac'):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    did_leave = trace_particles_starting_on_curve_gpu(
        ma, bfield, nparticles, tmax=1e-2, seed=1, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
        Ekin=5000*ONE_EV, umin=-1, umax=+1, comm=comm_world,
        phis=phis, tol=1e-9,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)
    print(did_leave)
    t2 = time.time()
    proc0_print(f"Time for particle tracing={t2-t1:.3f}s.", flush=True)
    # if comm_world is None or comm_world.rank == 0:
    #     # particles_to_vtk(gc_tys, OUT_DIR + f'particles_{label}_{mode}')
    #     plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{label}_loss.png', mark_lost=True)
    #     plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{label}.png', mark_lost=False)


proc0_print('Error in B', bsh.estimate_error_B(1000), flush=True)
proc0_print('Error in AbsB', bsh.estimate_error_GradAbsB(1000), flush=True)
trace_particles(bsh, 'bsh', 'gc_vac')
# trace_particles(bsh, 'bsh', 'full')
# trace_particles(bs, 'bs', 'gc')

proc0_print("End of 1_Simple/tracing_particle.py")
proc0_print("====================================")