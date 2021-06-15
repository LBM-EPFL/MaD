import os
import sys
import numpy as np
from math import floor, ceil, sqrt
from scipy.signal import convolve

class PDB(object):
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.coords = []
        self.info = []

        if not os.path.exists(self.pdb_file):
            print("PDB> File not found: %s"%self.pdb_file)
            sys.exit(1)

        '''
        Load Coordinate sections of a PDB file as of PDB version 3.30

        COLUMNS        DATA  TYPE    FIELD        DEFINITION
        -------------------------------------------------------------------------------------
         1 -  6        Record name   "ATOM  "
         7 - 11        Integer       serial       Atom  serial number.
        13 - 16        Atom          name         Atom name.
        17             Character     altLoc       Alternate location indicator.
        18 - 20        Residue name  resName      Residue name.
        22             Character     chainID      Chain identifier.
        23 - 26        Integer       resSeq       Residue sequence number.
        27             AChar         iCode        Code for insertion of residues.
        31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
        39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
        47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
        55 - 60        Real(6.2)     occupancy    Occupancy.
        61 - 66        Real(6.2)     tempFactor   Temperature  factor.
        77 - 78        LString(2)    element      Element symbol, right-justified.
        79 - 80        LString(2)    charge       Charge  on the atom.
        '''
        self.CA_idx = []
        self.BB_idx = []
        c = 0
        with open(self.pdb_file, 'r') as pdb:
            for line in pdb:
                line_type = line[0:6].strip()
                if line_type in ["ATOM", "HETATM"]:
                    try:
                        at_num = int(line[6:11].strip())
                        at_name = line[12:16].strip()
                        res_name = line[17:20]
                        chain_id = line[21]
                        res_num = int(line[22:26].strip())
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        element_symbol = line[76:78].strip()
                    except Exception:
                        # print("PDB> Error while reading line, skipping it: \n     > %s"%line)
                        pass

                    self.info.append([at_num, at_name, res_name, chain_id, res_num, element_symbol, line_type])
                    self.coords.append([x, y, z])
                    if at_name == "CA":
                        self.CA_idx.append(c)
                    if at_name in ["C", "CA", "N", "O"]:
                        self.BB_idx.append(c)
                    c += 1

        self.coords = np.array(self.coords)
        self.CA_idx = tuple(self.CA_idx)
        self.n_atoms = len(self.coords)
        self.n_CA = len(self.CA_idx)

        # Extent of structure
        self.minx = np.amin(self.coords[:, 0])
        self.maxx = np.amax(self.coords[:, 0])
        self.miny = np.amin(self.coords[:, 1])
        self.maxy = np.amax(self.coords[:, 1])
        self.minz = np.amin(self.coords[:, 2])
        self.maxz = np.amax(self.coords[:, 2])

    def write_pdb(self, outname):
        occupancy = 1.0
        tempFactor = 0.0
        fout = open(outname, "w")
        for i in range(self.n_atoms):
            # Check for atom type: if 4 letters long, start in column 13 otherwise no space.
            # Else, start in column 14. Differs from format convention v3.30 but always see it this way in files.
            if len(self.info[i][1]) == 4:
                line_model = "%-6s%5i %-4s %3s%2s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s"
            else:
                line_model = "%-6s%5i  %-3s %3s%2s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s"
            line = line_model%(self.info[i][6], self.info[i][0], self.info[i][1], self.info[i][2], self.info[i][3], self.info[i][4], self.coords[i][0], self.coords[i][1], self.coords[i][2], occupancy, tempFactor, self.info[i][5])

            fout.write(line+"\n")
        fout.close()

    def rgyr(self):
        rgyr = np.sqrt(np.sum(np.sum((self.coords - self.center())**2, axis=1) / self.coords.shape[0]))
        return rgyr

    ####################
    # PDB manipulation #
    ####################
    def get_coords(self):
        return self.coords

    def set_coords(self, coords):
        self.coords = coords.copy()

    def rotate_atoms(self, rot_mat):
        self.coords = np.dot(self.coords, rot_mat)

    def translate_atoms(self, trans_vec):
        self.coords += np.array(trans_vec)

    def get_rmsd_with(self, pdb):
        dist_sq = np.square(self.coords - pdb.coords)
        return np.sqrt(np.sum(dist_sq, axis=(0,1)) / dist_sq.shape[0])

    def get_rmsdCA_with(self, pdb):
        if not len(self.CA_idx):
            print("PDB> No alpha carbons detected; returning all-atom RMSD instead.")
            return self.get_rmsd_with(pdb)
        dist_sq = np.square(self.coords[self.CA_idx, :] - pdb.coords[pdb.CA_idx, :])
        return np.sqrt(np.sum(dist_sq, axis=(0,1)) / dist_sq.shape[0])


    ###################
    # Density related #
    ###################
    # Get density grid
    def structure_to_density(self, resolution, voxelsp, isovalue=0.0, pad=0, outname=""):
        # Interpolate pdb to grid + box size + origin coordinates
        grid_prot, pxb, pyb, pzb, minx, miny, minz = self.interpolate_to_grid_massweighted(voxelsp, pad=pad)

        # Margin for convoluted box
        margin = 2 + pad

        # Sigma
        # > Divide by pi*sqrt(2): it is a factor that affects the voxelsp of the gaussian.
        #   Same factor as in chimera. Their explanation in documentation:
        #       Fourier transform (FT) of the distribution fall to 1/e of its maximum value at wavenumber 1/resolution
        # > Divide by voxel spacing to make sure kernel relates to structure space
        # > Truncate values to 3 sigma
        sig = resolution / (np.pi*sqrt(2)) / voxelsp
        r = int(ceil(3.0 * sig))  # truncate at 3 sigma

        # Make Gaussian kernel
        z,y,x = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        h = np.exp( -(x*x + y*y + z*z) / (2. * sig**2) )
        kernel = h / h.sum()

        # Make density map at given resolution from structure grid
        grid_prot = np.reshape(grid_prot, (pxb, pyb, pzb), order='F')
        grid_dens = convolve(grid_prot, kernel).astype(np.float32)

        # Origin and size of convoluted map
        dxi = minx - (r + margin) * voxelsp;
        dyi = miny - (r + margin) * voxelsp;
        dzi = minz - (r + margin) * voxelsp;

        # Bring map in range [0,1] + apply isovalue threshold
        grid_dens = np.divide(grid_dens, np.amax(grid_dens))
        grid_dens[np.where(grid_dens < isovalue)] = 0

        if outname != "":
            extension = os.path.splitext(outname)[-1].lower()
            if extension in [".sit", ".situs"]:
                dxb, dyb, dzb = grid_dens.shape
                f_out = open(outname,"w")
                f_out.write("%f %f %f %f %i %i %i\n\n"%(voxelsp, dxi, dyi, dzi, dxb, dyb, dzb))
                voxi = 0
                for z in range(dzb):
                    for y in range(dyb):
                        for x in range(dxb):
                            if (voxi+1)%10 == 0 :
                                f_out.write("\n")
                            f_out.write("   %6.6f   "%grid_dens[x][y][z])
                            voxi += 1
                f_out.close()
            else:
                # assuming mrc/map
                import mrcfile
                with mrcfile.new(outname, overwrite=True) as mrc:
                    mrc.set_data(grid_dens.transpose(2,1,0).astype(np.float32))
                    xb, yb, zb = grid_dens.shape

                    mrc.mode = 2

                    mrc.header.mx = xb
                    mrc.header.my = yb
                    mrc.header.mz = zb

                    mrc.header.nxstart = 0
                    mrc.header.nystart = 0
                    mrc.header.nzstart = 0
                    mrc.header.origin.x = dxi
                    mrc.header.origin.y = dyi
                    mrc.header.origin.z = dzi

                    mrc.header.cella.x = xb * voxelsp
                    mrc.header.cella.y = yb * voxelsp
                    mrc.header.cella.z = zb * voxelsp

                    mrc.header.mapc = 1
                    mrc.header.mapr = 2
                    mrc.header.maps = 3

        return grid_dens.astype(np.float32), dxi, dyi, dzi


    #################
    # Interpolation #
    #################

    def interpolate_to_grid_massweighted(self, voxelsp, pad=0):
        # Convert 3D idx to 1D
        def idz(kx, ky, k, j, i):
            return kx * ky * k + kx * j + i

        mass_dict = {"H" : 1.00797, "BE": 9.01218, "C" : 12.011, "N" : 14.0067, "O" : 15.9994, "F": 18.998403, "S" : 32.06, "P" : 30.97376,
                     "MG": 24.305, "CL": 35.453, "K": 39.0983, "CA": 40.078, "MN": 54.9380, "FE": 55.847, "NI": 58.70, "CU": 63.546, "ZN": 65.38, "SE": 78.96}

        # To build the density map, we need the XYZ coordinates and mass of for each atom from PDB
        info_for_dens = []
        element_list = [x[-2].upper() for x in self.info]
        for xyz, elem in zip(self.coords, element_list):
            x, y, z = xyz
            if elem in mass_dict.keys():
                mass = mass_dict[elem]
            else:
                print("PDB> (dens) Element %s not in dict. Using mass of carbon."%elem)
                mass = mass_dict["C"]  # default value
            info_for_dens.append([x, y, z, mass])
        info_for_dens = np.array(info_for_dens)

        # Extent of structure
        minx = np.amin(info_for_dens[:, 0])
        maxx = np.amax(info_for_dens[:, 0])
        miny = np.amin(info_for_dens[:, 1])
        maxy = np.amax(info_for_dens[:, 1])
        minz = np.amin(info_for_dens[:, 2])
        maxz = np.amax(info_for_dens[:, 2])

        # Lattice into register with origin

        minx = voxelsp * floor(minx / voxelsp)
        maxx = voxelsp * ceil(maxx / voxelsp)
        miny = voxelsp * floor(miny / voxelsp)
        maxy = voxelsp * ceil(maxy / voxelsp)
        minz = voxelsp * floor(minz / voxelsp)
        maxz = voxelsp * ceil(maxz / voxelsp)

        # Size of box in each direction to contain structure
        margin = 2 + pad
        pxb = ceil((maxx - minx) / voxelsp) + 2 * margin + 1
        pyb = ceil((maxy - miny) / voxelsp) + 2 * margin + 1
        pzb = ceil((maxz - minz) / voxelsp) + 2 * margin + 1

        nvox_prot = pxb * pyb * pzb
        grid_prot = np.zeros((nvox_prot, 1))

        # Trilinear interpolation of structure to box
        for i in range(len(info_for_dens)):
            # position within grid
            gx = margin + (info_for_dens[i][0] - minx) / voxelsp
            gy = margin + (info_for_dens[i][1] - miny) / voxelsp
            gz = margin + (info_for_dens[i][2] - minz) / voxelsp

            # XYZ coords surrounding atom position
            x0 = floor(gx)
            y0 = floor(gy)
            z0 = floor(gz)
            x1 = x0 + 1;
            y1 = y0 + 1;
            z1 = z0 + 1;

            # interpolate
            a = x1 - gx
            b = y1 - gy
            c = z1 - gz
            grid_prot[idz(pxb, pyb, z0, y0, x0)] += info_for_dens[i][3] * a * b * c
            grid_prot[idz(pxb, pyb, z1, y0, x0)] += info_for_dens[i][3] * a * b * (1 - c)
            grid_prot[idz(pxb, pyb, z0, y1, x0)] += info_for_dens[i][3] * a * (1 - b) * c
            grid_prot[idz(pxb, pyb, z0, y0, x1)] += info_for_dens[i][3] * (1 - a) * b * c
            grid_prot[idz(pxb, pyb, z1, y1, x0)] += info_for_dens[i][3] * a * (1 - b) * (1 - c)
            grid_prot[idz(pxb, pyb, z0, y1, x1)] += info_for_dens[i][3] * (1 - a) * (1 - b) * c
            grid_prot[idz(pxb, pyb, z1, y0, x1)] += info_for_dens[i][3] * (1 - a) * b * (1 - c)
            grid_prot[idz(pxb, pyb, z1, y1, x1)] += info_for_dens[i][3] * (1 - a) * (1 - b) * (1 - c)

        grid_prot = np.divide(grid_prot, np.amax(grid_prot))

        return grid_prot, pxb, pyb, pzb, minx, miny, minz
