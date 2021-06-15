import os
import numpy as np
from skimage.feature import peak_local_max
from .DensityFeature import DensityFeature

class Detector(object):

    def __init__(self):
        # Detecting stats
        self.lowdensity = 0
        self.lowcontrast = 0
        self.saddlepoint = 0
        self.lowratio = 0
        self.largeoffset = 0
        self.badhessian = 0


    def find_anchors(self, ms, outname=""):
        if outname != "" and not os.path.exists(os.path.split(outname)[0]):
            print("Detector> WARNING: if writing files, a valid outname must be specified.")
            print("          Specified: %s"%outname)
            outname = ""

        print("MaD> Finding anchors in %s... "%ms.name)
        df_list = []
        for o, grid in enumerate(ms.map_space):
            xb, yb, zb = grid.shape
            border_shift = 12
            peaks = peak_local_max(grid, exclude_border=border_shift, threshold_abs=5e-2)
            for peak in peaks:
                # check keypoint
                is_good, coord, subcoord = self.check_localize(grid, peak)
                if not is_good:
                    continue
                val = grid[tuple(peak)]

                # Good keypoint, add it and break scale search
                df = DensityFeature()
                ox, oy, oz = coord
                subx, suby, subz = subcoord
                index = len(df_list)
                map_coords = self.get_coord_in_ref_map(ox, oy, oz, ms.xi, ms.yi, ms.zi, ms.voxelsp_list[o])
                subv_map_coords = self.get_coord_in_ref_map(subx, suby, subz, ms.xi, ms.yi, ms.zi, ms.voxelsp_list[o])
                df.set_detector_info(index, o, [ox,oy,oz], map_coords, subv_map_coords, val)
                df_list.append(df)

        if outname:
            self.write_df_to_file(df_list, outname+"_data.txt")
            self.write_df_to_pdb(df_list, outname+".pdb")
        return df_list


    def check_localize(self, grid, oricoord):
        x, y, z = oricoord

        # Interpolate position of max in space
        # > Compute Hessian with finite differences
        #       XX  XY  XZ
        # H = ( XY  YY  YZ )
        #       XZ  YZ  ZZ
        maxOffset = 0.6
        n_iter = 0
        sublocalized = False

        while n_iter < 5:
            # Hessian
            xx = grid[x-1,y,z] + grid[x+1,y,z] - 2*grid[x,y,z]
            yy = grid[x,y-1,z] + grid[x,y+1,z] - 2*grid[x,y,z]
            zz = grid[x,y,z-1] + grid[x,y,z+1] - 2*grid[x,y,z]
            xy = 0.25 * ( ( grid[x+1,y+1,z] - grid[x+1,y-1,z] ) - (grid[x-1,y+1,z] - grid[x-1,y-1,z]) )
            xz = 0.25 * ( ( grid[x+1,y,z+1] - grid[x+1,y,z-1] ) - (grid[x-1,y,z+1] - grid[x-1,y,z-1]) )
            yz = 0.25 * ( ( grid[x,y+1,z+1] - grid[x,y+1,z-1] ) - (grid[x,y-1,z+1] - grid[x,y-1,z-1]) )
            H = np.array([ [xx, xy, xz], [xy, yy, yz], [xz, yz, zz] ])

            # Gradient
            gX = 0.5 * ( grid[x+1,y,z] - grid[x-1,y,z] )
            gY = 0.5 * ( grid[x,y+1,z] - grid[x,y-1,z] )
            gZ = 0.5 * ( grid[x,y,z+1] - grid[x,y,z-1] )
            G = np.array([gX, gY, gZ])

            try:
                Hinv = np.linalg.inv(H)
            except Exception:
                return False, oricoord, oricoord

            offset = - np.dot(Hinv, G)
            if np.all(np.abs(offset) < maxOffset):
                sublocalized = True
                break
            else:
                ox, oy, oz = offset
                if ox < - maxOffset and x - 1 > 0: x -= 1
                elif ox > maxOffset and x + 1 < grid.shape[0] - 1: x += 1
                if oy < - maxOffset and y - 1 > 0: y -= 1
                elif oy > maxOffset and y + 1 < grid.shape[1] - 1: y += 1
                if oz < - maxOffset and z - 1 > 0: z -= 1
                elif oz > maxOffset and z + 1 < grid.shape[2] - 1: z += 1
            n_iter += 1

        # Check+prepare anchor if it was sublocalized successfully
        ox, oy, oz = offset
        if sublocalized:
            eigvals = np.array(sorted(np.linalg.eigvals(H)))

            # Remove saddle points; as we only have maxima, it is enough to check for positive eigvals. They should all be negative.
            if np.any(eigvals > 0):
                return False, oricoord, oricoord

            # Check curvature
            # > disabled, didn't seem to be major differences for repeatable/non-repeatable points)
            # ratio = eigvals[2] / eigvals[0]
            # if ratio < 0.05:
            # if ratio < 0.01:
            #     return False, oricoord, oricoord

            # If still here, add computed offset
            fx = x + ox
            fy = y + oy
            fz = z + oz

            return True,  [x,y,z], [fx,fy,fz]
        else:
            return False, oricoord, oricoord


    def get_coord_in_ref_map(self, x, y, z, xi, yi, zi, voxsp):
        # Convert array coordinate to coordinates within the density map
        return np.array([x * voxsp + xi, y * voxsp + yi, z * voxsp + zi])


    #######
    # I/O #
    #######

    def write_df_to_file(self, df_list, outname):
        np.save(outname, df_list)


    def load_df_from_file(self, df_file):
        df_list = np.load(df_file, allow_pickle=True)
        print("Det> Loaded %i anchors."%len(df_list))
        return df_list


    def write_df_to_pdb(self, df_list, outname, save_regular=False):
        # Create dummy atoms for anchor vor visualization in VMD or similar
        # > Two chains:
        #   > 1. with anchors coordinates off-lattice used for all steps
        #   > 2. with anchors coordinates on the grid (save_regular=True)
        pdb_data = []

        for idx in range(len(df_list)):
            atom_type = 'O'
            atom_res = "SUB"
            atom_chain = 'A'

            tmp_atom = []
            tmp_atom.append(idx) #=atom_nb
            tmp_atom.append(atom_type)
            tmp_atom.append(atom_res)
            tmp_atom.append(atom_chain)
            tmp_atom.append(idx) #=res_nb
            tmp_atom.append(df_list[idx].subv_map_coords[0])
            tmp_atom.append(df_list[idx].subv_map_coords[1])
            tmp_atom.append(df_list[idx].subv_map_coords[2])
            pdb_data.append(tmp_atom)

        if save_regular:
            for idx in range(len(df_list)):
                atom_type = 'O'
                atom_res = "ORI"
                atom_chain = 'B'

                tmp_atom = []
                tmp_atom.append(idx) #=atom_nb
                tmp_atom.append(atom_type)
                tmp_atom.append(atom_res)
                tmp_atom.append(atom_chain)
                tmp_atom.append(idx) #=res_nb
                tmp_atom.append(df_list[idx].map_coords[0])
                tmp_atom.append(df_list[idx].map_coords[1])
                tmp_atom.append(df_list[idx].map_coords[2])
                pdb_data.append(tmp_atom)

        with open(outname,"w") as f:
            for x in pdb_data:
                line_for_pdb ='ATOM%7i%5s%5s%1s%4i    %8.3f%8.3f%8.3f\n'\
                    %(int(x[0]),x[1],x[2],x[3],int(x[4]),float(x[5]),float(x[6]),float(x[7]))
                f.write(line_for_pdb)
