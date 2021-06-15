import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

from .PDB import PDB
from .Dmap import Dmap
from .math_utils import euler_rod_mat, unit_vector

def move_structure(original_struct, t=None, a=0.375, b=1.735, c=2.452, suffix=""):
    '''
    Moves a subunit already placed in the reference map to the origin
    by subtracting its geometrical center and rotates it.
    Default values for a, b, c has been chosen randomly
    '''
    moved_struct = original_struct.replace(".pdb", "_moved%s.pdb"%suffix)

    pdb = PDB(original_struct)

    pdb.rotate_atoms(euler_rod_mat([1,0,0], a))
    pdb.rotate_atoms(euler_rod_mat([0,1,0], b))
    pdb.rotate_atoms(euler_rod_mat([0,0,1], c))
    coords = pdb.get_coords()
    if t is None:
        pdb.translate_atoms(-np.mean(coords, axis=0))
    else:
        pdb.translate_atoms(t)
    pdb.write_pdb(moved_struct)

    return moved_struct

def move_copy_structure(original_struct, moved_struct, transform=False, t=[150, 0, 0], a=0.375, b=1.735, c=2.452):
    '''
    Copies a structure.

    If transform is true:
    Moves a structure to the origin and rotate
    Default values for a, b, c has been chosen randomly
    '''
    pdb = PDB(original_struct)

    if transform:
        pdb.rotate_atoms(euler_rod_mat([1,0,0], a))
        pdb.rotate_atoms(euler_rod_mat([0,1,0], b))
        pdb.rotate_atoms(euler_rod_mat([0,0,1], c))
        coords = pdb.get_coords()

        # move to origin
        pdb.translate_atoms(-np.mean(coords, axis=0))

        # further translate, useful to check corresponding anchors
        if len(t):
            pdb.translate_atoms(t)

    # write moved structure in e.g. results folder/initial files
    pdb.write_pdb(moved_struct)

    return moved_struct

def refine_pdb(dmap, pdb, n_steps=500, max_step_size=0.5, min_step_size=0.01, idx=-1):
    # Get map information
    voxelsp = dmap.voxsp
    sx, sy, sz = dmap.grid3d.shape
    ox, oy, oz = dmap.xi, dmap.yi, dmap.zi

    # Get pdb information
    pdb_init_coords = pdb.coords.copy()
    pdb_center = np.mean(pdb_init_coords, axis=0)
    max_dist_from_center = np.amax(np.linalg.norm(pdb_init_coords - pdb_center, axis=1))

    # Base transformation
    trans = [0,0,0]
    rot_mat = np.identity(3)

    # Set up interpolator
    # > Gradient axis needs to be changed from (3,nx,ny,nz) to (nx,ny,nz,3))
    # > Gradient vectors are associated to their real-space coordinates for easy interpolation from atom coordinates
    px = np.arange(ox, voxelsp * sx + ox, voxelsp)[:dmap.xb]  # limit size to xb if rounding errors add one more point
    py = np.arange(oy, voxelsp * sy + oy, voxelsp)[:dmap.yb]
    pz = np.arange(oz, voxelsp * sz + oz, voxelsp)[:dmap.zb]

    rgi = RGI(points=[px, py, pz], values=np.moveaxis(np.array(np.gradient(dmap.grid3d)), 0, -1))

    # Refinement loop
    batch_size = 4
    step_size = max_step_size
    step_batch_init = [np.zeros(3), np.identity(3), 0]
    step_batch = step_batch_init.copy()
    prev_atom_transf = pdb_init_coords.copy()
    converged = False
    for step in range(n_steps):
        # Re-initialize transformation
        pdb.set_coords(pdb_init_coords)

        # Apply current transformation to protein
        pdb.translate_atoms(-pdb_center)
        pdb.rotate_atoms(rot_mat)
        pdb.translate_atoms(pdb_center + trans)
        if np.any(np.isnan(pdb.coords)):
            return np.nan, False, step

        # Get atoms that are within bounds
        in_map_mask = np.where((pdb.coords[:, 0] > ox) & (pdb.coords[:, 0] < ox + sx * voxelsp - voxelsp) &\
                               (pdb.coords[:, 1] > oy) & (pdb.coords[:, 1] < oy + sy * voxelsp - voxelsp) &\
                               (pdb.coords[:, 2] > oz) & (pdb.coords[:, 2] < oz + sz * voxelsp - voxelsp))

        # Get gradient from map at atom positions
        atom_gradients = rgi(pdb.coords[in_map_mask])

        # > Translation steps (even number of steps)
        if not step % 2:
            # Get force from density gradient
            step_trans = unit_vector(np.sum(atom_gradients, axis=0)) * step_size

            # Update transformation for batch and overall
            step_batch[0] += step_trans
            trans += step_trans
            pdb.translate_atoms(step_trans)

        # > Rotation steps (odd number of steps)
        else:
            # Get torque
            centered_coords = pdb.coords - pdb_center
            axis_torque = np.sum(np.cross(atom_gradients, centered_coords[in_map_mask]), axis=0)
            axis_torque = unit_vector(axis_torque)

            # Get angle so that displacement of farthest atom is step_size
            # > Formula for distance around a point is r * angle = d
            # > r is max_dist_from_center, d step_size in our case, angle is in rad
            angle = step_size / max_dist_from_center
            step_rot_mat = euler_rod_mat(axis_torque, angle)

            # Apply new rotation: go to pdb_center first
            pdb.translate_atoms(-1 * pdb_center - trans)
            pdb.rotate_atoms(step_rot_mat)
            pdb.translate_atoms(pdb_center + trans)

            # Update transformation for batch and overall
            step_batch[1] = step_batch[1].dot(step_rot_mat)
            rot_mat = rot_mat.dot(step_rot_mat)

        # Try to update step size if batch is complete
        step_batch[2] += 1
        if step_batch[2] == batch_size:
            max_norm = np.amax(np.linalg.norm(prev_atom_transf - pdb.coords, axis=1))
            if max_norm < step_size:
                step_size *= 0.5
            step_batch = step_batch_init.copy()
            prev_atom_transf = pdb.coords.copy()

        # Check convergence
        if step_size < min_step_size:
            converged = True
            break

    # RMSD before-after fitting
    if len(pdb.CA_idx):
        dist_sq = np.square(pdb.coords[pdb.CA_idx, :] - pdb_init_coords[pdb.CA_idx, :])
    else:
        dist_sq = np.square(pdb.coords - pdb_init_coords)
    rmsd_beforeAfter = np.sqrt(np.sum(dist_sq, axis=(0,1)) / dist_sq.shape[0])

    return rmsd_beforeAfter, converged, step

def get_overlap(g1, g2, voxsp, isovalue=1e-8):
    # Origins and extent of overlap map
    grid1, xi1, yi1, zi1 = g1
    grid2, xi2, yi2, zi2 = g2

    xb1, yb1, zb1 = grid1.shape
    xb2, yb2, zb2 = grid2.shape

    grid1[grid1 < isovalue] = 0
    grid2[grid2 < isovalue] = 0

    xi1 /= voxsp
    yi1 /= voxsp
    zi1 /= voxsp
    xi2 /= voxsp
    yi2 /= voxsp
    zi2 /= voxsp

    # lower box bounds
    # X
    if xi1 > xi2:
        xmin1 = int(0)
        xmin2 = int(round(xi1-xi2))
    elif xi1 < xi2:
        xmin1 = int(round(xi2 - xi1))
        xmin2 = int(0)
    else:
        xmin1, xmin2 = int(0), int(0)

    #Y
    if yi1 > yi2:
        ymin1 = int(0)
        ymin2 = int(round(yi1 - yi2))
    elif yi1 < yi2:
        ymin1 = int(round(yi2 - yi1))
        ymin2 = int(0)
    else:
        ymin1, ymin2 = int(0), int(0)

    #Z
    if zi1 > zi2:
        zmin1 = int(0)
        zmin2 = int(round(zi1 - zi2))
    elif zi1 < zi2:
        zmin1 = int(round(zi2 - zi1))
        zmin2 = int(0)
    else:
        zmin1, zmin2 = int(0), int(0)

    # upper box bouds
    #X
    if xi1 + xb1 > xi2 + xb2 :
        xmax1 = int(round(xi2 + xb2 - xi1))
        xmax2 = int(round(xb2))
    elif xi1 + xb1 < xi2 + xb2 :
        xmax1 = int(round(xb1))
        xmax2 = int(round(xi1 + xb1 - xi2))
    else:
        xmax1, xmax2 = int(round(xb1)), int(round(xb2))

    #Y
    if yi1 + yb1 > yi2 + yb2 :
        ymax1 = int(round(yi2 + yb2 - yi1))
        ymax2 = int(round(yb2))
    elif yi1 + yb1 < yi2 + yb2 :
        ymax1 = int(round(yb1))
        ymax2 = int(round(yi1 + yb1 - yi2))
    else:
        ymax1, ymax2 = int(round(yb1)), int(round(yb2))

    #Z
    if zi1 + zb1 > zi2 + zb2 :
        zmax1 = int(round(zi2 + zb2  - zi1))
        zmax2 = int(round(zb2))
    elif zi1 + zb1 < zi2 + zb2 :
        zmax1 = int(round(zb1))
        zmax2 = int(round(zi1 + zb1  - zi2))
    else:
        zmax1, zmax2 = int(round(zb1)), int(round(zb2))

    if xmax1-xmin1 < 0 or ymax1-ymin1 < 0 or zmax1-zmin1 < 0 \
       or xmax1-xmin1 < 0 or ymax1-ymin1 < 0 or zmax1-zmin1 < 0:
        return 0

    # Select common box
    map1 = grid1[xmin1:xmax1, ymin1:ymax1, zmin1:zmax1].copy()
    map2 = grid2[xmin2:xmax2, ymin2:ymax2, zmin2:zmax2].copy()

    # Reshape to 1D
    map1_1d = np.reshape(map1, map1.shape[0]*map1.shape[1]*map1.shape[2], order='F')
    map2_1d = np.reshape(map2, map2.shape[0]*map2.shape[1]*map2.shape[2], order='F')

    common = len(map1_1d[np.where((map1_1d > 0) & (map2_1d > 0))])
    m1_vals = np.count_nonzero(grid1[np.where(grid1 > 0)])
    if m1_vals == 0:
        return 0
    return common / m1_vals