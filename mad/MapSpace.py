import os
import sys
import mrcfile
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_laplace
from scipy.interpolate import RegularGridInterpolator as RGI

from .PDB import PDB

class MapSpace(object):
    def __init__(self, structure_file, resolution=0, voxelsp=0, isovalue=0.0, map_padding=9, oct_mode="both", sig_init=2, sig_presmooth=1):
        """

        Parameters
        ----------
        structure_file : str
            filename of input structure. Can be either a pdb, situs or sit file.
        resolution : str, optional
            Mandatory when providing a PDB file. This will be the resolution of the simulated map.
        voxelsp : float, optional
            Mandatory when providing a PDB file. This will be the voxel spacing of the simulated map.
        isovalue : float, optional
            Density values below this value will be set to 0.
            For a PDB structure, the isovalue is applied AFTER bringing the map range to [0,1] due to the simulation process
            For situs maps, the isovalue is applied BEFORE bringing the map range to [0,1]
            The default is 0. (all values are kept)
        map_padding : int, optional
            Applies zero-padding to map. This can help if the structure is too close to map borders.
            This can prevent proper descriptor building.
            The default is 0.

        Returns
        -------
        None.

        """
        self.structure_file = structure_file  # Situs/sit or PDB file
        self.isovalue = isovalue
        self.map_padding = map_padding
        self.PDB_mode = False
        self.name = os.path.splitext(os.path.split(structure_file)[-1])[0]
        self.sig_init = sig_init
        self.sig_presmooth = sig_presmooth
        self.oct_mode = oct_mode
        if oct_mode not in ["base", "up", "both"]:
            print("MaD> WARNING: #octave not set properly (%s), reverting to 'base'"%oct_mode)
            self.oct_mode = "base"

        # Check extension of structure
        # > Need to convert PDB to density grid before building
        self.ext = os.path.splitext(structure_file)[-1].lower()
        if self.ext == ".pdb":
            self.PDB_mode = True
            self.voxelsp = voxelsp
            if self.voxelsp == 0:
                print("MaD> ERROR: if providing a PDB, voxel spacing is mandatory")
                sys.exit(1)
            self.resolution = resolution
            if self.resolution == 0:
                print("MaD> ERROR: if providing a PDB, resolution is mandatory")
                sys.exit(1)
        elif self.ext not in [".situs", ".sit", ".map", ".mrc"]:
            print("MaD> ERROR: please provide a valid structure file (pdb, sit, situs, map or mrc format)")
            print(self.ext)
            sys.exit(1)

    def build_space(self):
        print("MaD> Building map space for %s..."%self.name)

        # Get density grid from structure file
        if self.PDB_mode:
            pdb = PDB(self.structure_file)
            grid, xi, yi, zi = pdb.structure_to_density(self.resolution, self.voxelsp, isovalue=self.isovalue)
            xb, yb, zb = grid.shape
        elif self.ext in [".situs", ".sit"]:
            sit = open(self.structure_file, "r")

            # Save header information: origin and dimensions
            # > remove end-of-line, any double spaces, then split into values
            header = sit.readline().replace("\n","").replace("  ","").split(" ")
            sit.readline() # there is always a blank line between header and voxel values
            grid1d = np.fromstring(sit.read(), sep='    ')
            sit.close()

            # Save header information in attribute variables
            self.voxelsp, xi, yi, zi = [float(x) for x in header[:4]]
            xb, yb, zb = [int(x) for x in header[4:]]

            # Save voxel data (omits space between values)
            grid1d[np.where(grid1d < self.isovalue)] = 0
            grid = np.reshape(grid1d, (xb, yb, zb), order='F')

            # Normalize to 1.0
            grid = grid / np.amax(grid).astype(np.float32)
        else: # mrc / map format
            with mrcfile.open(self.structure_file) as mrc:
                axis_order = [mrc.header.mapc-1, mrc.header.mapr-1, mrc.header.maps-1]
                self.voxelsp = mrc.voxel_size.x # when you finally find a proper documentation

                # Can't find consistent documentation for this. In EMDB, map files have nxstart/etc
                # MRC tends to have origin instead.
                if np.all([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart]):
                    origin = np.array([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=int)
                    xi, yi, zi = [origin[axis] * self.voxsp for axis in axis_order]
                else:
                    origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=int)
                    xi, yi, zi = [origin[axis]  for axis in axis_order]
                boxdim = np.array([mrc.header.mx, mrc.header.my, mrc.header.mz], dtype=int)

                xb, yb, zb = [boxdim[axis] for axis in axis_order]
                grid = np.transpose(mrc.data.copy(), axis_order[::-1]).astype(np.float32)
                grid[np.where(grid < self.isovalue)] = 0

        # Pad map, update map origin + size
        if self.map_padding:
            grid = np.pad(grid, self.map_padding, mode="constant")

            xi -= self.map_padding * self.voxelsp
            yi -= self.map_padding * self.voxelsp
            zi -= self.map_padding * self.voxelsp
            xb += 2 * self.map_padding
            yb += 2 * self.map_padding
            zb += 2 * self.map_padding

        # Map origin; remains the same over scale-space
        self.xi = xi
        self.yi = yi
        self.zi = zi

        if self.oct_mode in ["up", "both"]:
             # Upsampled map
            # > For higher precision / repeatability (as stated in SIFT, forstner, SURF, etc...)
            #   and higher accuracy due to key points being localised on a grid of higher granularity
            # > Smoothed slightly to cover interpolation artifacts
            rx = np.arange(0, xb, 1)
            ry = np.arange(0, yb, 1)
            rz = np.arange(0, zb, 1)
            rx_int = np.arange(0, xb-0.5, 0.5)
            ry_int = np.arange(0, yb-0.5, 0.5)
            rz_int = np.arange(0, zb-0.5, 0.5)
            if self.sig_presmooth:
                up_grid = gaussian_filter(self.interpn_so(rx, ry, rz, grid, rx_int, ry_int, rz_int), sigma=self.sig_presmooth).astype(np.float32)
            else:
                up_grid = self.interpn_so(rx, ry, rz, grid, rx_int, ry_int, rz_int).astype(np.float32)
            #~ print("MaD> Upsampled map from %s to %s..."%(str(grid.shape), str(up_grid.shape)))

        if self.oct_mode == "both":
            fgrid_list = [up_grid, grid]
            self.voxelsp_list = [self.voxelsp / 2, self.voxelsp]
            sigma_list = [self.sig_init, self.sig_init]
        elif self.oct_mode == "up":
            fgrid_list = [up_grid]
            self.voxelsp_list = [self.voxelsp / 2]
            sigma_list = [self.sig_init]
        elif self.oct_mode == "base":
            fgrid_list = [grid]
            self.voxelsp_list = [self.voxelsp]
            sigma_list = [self.sig_init]
        else:
            print("MaD> ERROR in octave mode %s"%self.oct_mode)
            sys.exit(1)

        # Kept to check original voxel values of solution-generating anchors
        self.grid_list = fgrid_list

        # scale-normalized LoG filtering
        self.map_space = []
        for o, fgrid in enumerate(fgrid_list):
            log_g = -1 * gaussian_laplace(fgrid, sigma=sigma_list[o]) * sigma_list[o]**2
            log_g[log_g < 0] = 0.0
            self.map_space.append(log_g)
        #~ self.map_space = np.array(self.map_space, dtype=object)
        #~ self.map_space = np.array(self.map_space, dtype=object)

        # Spaces for Descriptor: Gaussian filtered, density gradient grid, and primer for interpolation for descriptors
        self.gauss_list = []
        self.grad_list = []
        self.rgi_space = []
        for o, fgrid in enumerate(fgrid_list):
            self.gauss_list.append(gaussian_filter(fgrid, self.sig_init))
            sx,sy,sz = fgrid.shape
            points_x = np.arange(0,sx)
            points_y = np.arange(0,sy)
            points_z = np.arange(0,sz)
            self.grad_list.append(np.moveaxis(np.array(np.gradient(self.gauss_list[-1])), 0, -1))
            # self.grad_list.append(np.moveaxis(np.array(np.gradient(fgrid)), 0, -1))
            self.rgi_space.append(RGI(points=[points_x, points_y, points_z], values=self.grad_list[-1], method="nearest"))

    def interpn_so(self, *args, **kw):
        """Interpolation on N-D.

        ai = interpn(x, y, z, ..., a, xi, yi, zi, ...)
        where the arrays x, y, z, ... define a rectangular grid
        and a.shape == (len(x), len(y), len(z), ...)

        NOTE: Taken from stackoverflow; question #13333265 "Vector/arrqay output from 'scipy.ndimage.map_coordiantes';
                answer from user 'pv.' on Nov. 11 2012
        """
        method = kw.pop('method', 'cubic')

        if kw:
            raise ValueError("Unknown arguments: " % kw.keys())
        nd = (len(args)-1)//2

        if len(args) != 2*nd+1:
            raise ValueError("Wrong number of arguments")
        q = args[:nd]
        qi = args[nd+1:]
        a = args[nd]
        for j in range(nd):
            a = interp1d(q[j], a, axis=j, kind=method)(qi[j])
        return a
