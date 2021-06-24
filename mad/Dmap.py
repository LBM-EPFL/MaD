import os
import sys
import numpy as np
import mrcfile

class Dmap(object):
    def __init__(self, map_name, isovalue=0.0, normalize=True, pad=0):
        # read map based on type
        # > save from header box dimensiosn + origin + voxel spacing
        # > only supports saving to .sit format because i don't want to deal with headers
        if os.path.isfile(map_name):
            ext = os.path.splitext(map_name)[-1]
            if ext.lower() in [".sit", ".situs"]:
                with open(map_name, "r") as f:
                    # Save header information: origin and dimensions
                    # > remove end-of-line, any double spaces, then split into values
                    header = f.readline().replace("\n","").replace("  ","").split(" ")
                    f.readline() # there is always a blank line between header and voxel values
                    self.voxsp, self.xi, self.yi, self.zi = [float(x) for x in header[:4]]
                    self.xb, self.yb, self.zb = [int(x) for x in header[4:]]

                    # Save voxel data (omits space between values)
                    self.grid1d = np.fromstring(f.read(), sep='    ').astype(np.float32)
                    self.grid3d = np.reshape(self.grid1d, (self.xb, self.yb, self.zb), order='F')

            elif ext.lower() in [".map", ".mrc"]:
                with mrcfile.open(map_name) as mrc:
                    self.axis_order = [mrc.header.mapc-1, mrc.header.mapr-1, mrc.header.maps-1]
                    self.voxsp = mrc.voxel_size.x # when you finally find a proper documentation

                    # Can't find consistent documentation for this; some use nstarts, but Chimera saves MRC with "origin"
                    # > docs of mrcfile has no trace of "origin" in header info.
                    if np.all([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart]):
                        origin = np.array([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=int)
                        self.xi, self.yi, self.zi = [origin[axis] * self.voxsp for axis in self.axis_order]
                    else:

                        origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=int)
                        self.xi, self.yi, self.zi = [origin[axis]  for axis in self.axis_order]
                    boxdim = np.array([mrc.header.mx, mrc.header.my, mrc.header.mz], dtype=int)

                    self.xb, self.yb, self.zb = [boxdim[axis] for axis in self.axis_order]
                    self.grid3d = np.transpose(mrc.data.copy(), self.axis_order[::-1])
            else:
                print("Dmap> ERROR: incompatible extension for map %s"%map_name)
                return
        else:
            print("Dmap> ERROR: file %s not found"%map_name)
            sys.exit(1)
        if np.amax(self.grid3d > isovalue):
            self.grid3d[self.grid3d < isovalue] = 0
        else:
            print("Dmap> WARNING: asked isovalue is larger than maximum density found in file (%f). Considering isovalue=0"%np.amax(self.grid3d))
            self.grid3d[self.grid3d < 0] = 0

        if pad:
            self.grid3d = np.pad(self.grid3d, pad, mode="constant")
            self.xi -= pad*self.voxsp
            self.yi -= pad*self.voxsp
            self.zi -= pad*self.voxsp
            self.xb, self.yb, self.zb = self.grid3d.shape

        # Normalize to 1.0
        if np.isclose(np.amax(self.grid3d), 0):
            print("Dmap> WARNING: Max value in map is 0")
        if normalize:
            self.grid3d = self.grid3d / np.amax(self.grid3d)

        # Save remaining information
        self.map_name = map_name # whole filename
        self.name = map_name.split('/')[-1].split('.')[0] # "nickname" of the map. E.g: "/folder/7CAT.sit" => "7CAT"

    def reduce_void(self, zeros_padding=10):
        # Removes excess zero-valued voxels around structure and leaves only zeros_padding zero voxels around structure.

        # Get index of nonzero elements, take min and max in each direction to get map
        nz = np.nonzero(self.grid3d)
        minx, maxx = np.amin(nz[0]), np.amax(nz[0])
        miny, maxy = np.amin(nz[1]), np.amax(nz[1])
        minz, maxz = np.amin(nz[2]), np.amax(nz[2])

        # Adapt dimensions + reduce grid
        self.xi = self.xi + minx * self.voxsp
        self.yi = self.yi + miny * self.voxsp
        self.zi = self.zi + minz * self.voxsp
        self.grid3d = self.grid3d[minx:maxx+1, miny:maxy+1, minz:maxz+1]
        self.xb, self.yb, self.zb = self.grid3d.shape

        # Pad with a sane amount of zeros (default of 15 to account for descriptor building on structure border)
        self.pad_grid(zeros_padding)

    def pad_grid(self, pad):
        self.grid3d = np.pad(self.grid3d, pad, mode="constant")
        self.xi -= pad*self.voxsp
        self.yi -= pad*self.voxsp
        self.zi -= pad*self.voxsp
        self.xb, self.yb, self.zb = self.grid3d.shape

    def mask_with(self, mask_map):
        # This function assumes a voxel spacing of 1.0 for simplicity as they should have the same.
        # self has its content changed: if a voxel is 0 in mask_map, it is made 0 in self

        if not np.isclose(self.voxsp, mask_map.voxsp):
            print("ERROR: voxsp do not match! %f vs %f"%(self.voxsp, mask_map.voxsp))
            sys.exit(1)

        voxsp = self.voxsp

        # Origins and extent of overlap map
        xi1 = self.xi / voxsp
        yi1 = self.yi / voxsp
        zi1 = self.zi / voxsp
        xi2 = mask_map.xi / voxsp
        yi2 = mask_map.yi / voxsp
        zi2 = mask_map.zi / voxsp

        # Bounds and mapping
        sx = int(round(xi2 - xi1))
        sy = int(round(yi2 - yi1))
        sz = int(round(zi2 - zi1))
        if sx > 0:
            range_x_min = int(round(xi2 - xi1))
        else:
            range_x_min = 0

        if sy > 0:
            range_y_min = int(round(yi2 - yi1))
        else:
            range_y_min = 0

        if sz > 0:
            range_z_min = int(round(zi2 - zi1))
        else:
            range_z_min = 0

        range_x_max = min(self.xb, mask_map.xb + sx)
        range_y_max = min(self.yb, mask_map.yb + sy)
        range_z_max = min(self.zb, mask_map.zb + sz)

        # Nullify voxels before range of mask_map
        self.grid3d[:range_x_min, :, :] = 0
        self.grid3d[:, :range_y_min, :] = 0
        self.grid3d[:, :, :range_z_min] = 0
        self.grid3d[range_x_max:, :, :] = 0
        self.grid3d[:, range_y_max:, :] = 0
        self.grid3d[:, :, range_z_max:] = 0

        # Get zero voxels in mask map and apply in self
        common_mask = mask_map.grid3d[(range_x_min-sx):(range_x_max-sx), (range_y_min-sy):(range_y_max-sy), (range_z_min-sz):(range_z_max-sz)]
        zero_mask = np.where(common_mask < 1e-8)
        self.grid3d[range_x_min:range_x_max, range_y_min:range_y_max, range_z_min:range_z_max][zero_mask] = 0

    def get_CCC_with_grid(self, grid2, xi2, yi2, zi2, isovalue=0):
        # Origins and extent of overlap map
        xi1, yi1, zi1, xb1, yb1, zb1 = self.xi, self.yi, self.zi, self.xb, self.yb, self.zb
        xb2, yb2, zb2 = grid2.shape

        voxsp = self.voxsp
        grid1 = self.grid3d
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

        # Since the exp map has noise / additionnal densities not present in the simulated density, we eliminate it
        # by checking where density is 0 in the simulated grid
        # The reverse is also true sometimes, the fitted pdb has more info than the low-res map.
        # map1_1d[np.where(map2_1d) == 0] = 0
        # map2_1d[np.where(map1_1d) == 0] = 0
        olap = np.dot(map1_1d, map2_1d)
        norm_m1 = np.dot(map1_1d, map1_1d)
        norm_m2 = np.dot(map2_1d, map2_1d)
        denom = np.sqrt(norm_m1*norm_m2)

        ccc = olap / denom

        # print(ccc, np.cov(map1_1d, map2_1d)[0][0] / (np.std(map1_1d)**2))

        return ccc

    def get_CCC_with_dmap(self, m2, isovalue=0):
        # Origins and extent of overlap map
        xi1, yi1, zi1, xb1, yb1, zb1 = self.xi, self.yi, self.zi, self.xb, self.yb, self.zb
        xi2, yi2, zi2, xb2, yb2, zb2 = m2.xi, m2.yi, m2.zi, m2.xb, m2.yb, m2.zb

        if self.voxsp != m2.voxsp:
            print("ERROR: voxsp differ (%f vs %f)"%(self.voxsp, m2.voxsp))
        voxsp = self.voxsp
        grid1 = self.grid3d
        grid2 = m2.grid3d

        xi1 /= voxsp
        yi1 /= voxsp
        zi1 /= voxsp
        xi2 /= voxsp
        yi2 /= voxsp
        zi2 /= voxsp

        # Find common box between grids
        # lower box bounds
        # X
        if xi1 > xi2:
            xmin1 = int(0)
            xmin2 = int(round(xi1 - xi2))
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

        # If no !=0 voxels are found in both maps, don't bother going further
        nonzero_vox = min(np.count_nonzero(grid1 > isovalue), np.count_nonzero(grid2 > isovalue))
        common_vox = np.count_nonzero(map2_1d[(map2_1d > isovalue) & (map1_1d > isovalue)])
        if not common_vox or not nonzero_vox:
            return 0

        # Normalize: bigger map is scaled according to smaller map's nonzero voxels so that overlap is within [0,1]
        # > Best case scenario, all >0 voxels of map2 are also >0 in map1 => overlap = 1
        # > we don't want that other >0 voxels of map1 crash the score
        map1_1d /= np.linalg.norm(map1_1d[map2_1d>0])
        map2_1d /= np.linalg.norm(map2_1d[map1_1d>0])

        # Normalized binary maps => just use dot to get overlap
        ccc = np.dot(map1_1d, map2_1d)

        # scale overlap based on %voxels of smaller map that is in common with bigger map
        scale_fac = common_vox / nonzero_vox
        ccc *= scale_fac

        return ccc

    #########
    # Map I/O
    #########
    def write_to_sit(self, outname):
        f_out=open(outname,"w")
        print(">Dmap> Writing density map as %s"%(outname))
        f_out.write("%f %f %f %f %i %i %i\n\n"%(self.voxsp, self.xi, self.yi, self.zi, self.xb, self.yb, self.zb))

        voxi = 0
        for z in range(self.zb):
            for y in range(self.yb):
                for x in range(self.xb):
                    f_out.write("   %6.6f "%self.grid3d[x][y][z])
                    if (voxi+1)%10 == 0 :
                        f_out.write("\n")
                    voxi += 1
        f_out.close()

    def write_to_mrc(self, outname):
        with mrcfile.new(outname, overwrite=True) as mrc:
            mrc.set_data(self.grid3d.transpose(2,1,0))

            mrc.mode = 2

            mrc.header.mx = self.xb
            mrc.header.my = self.yb
            mrc.header.mz = self.zb

            mrc.header.nxstart = 0
            mrc.header.nystart = 0
            mrc.header.nzstart = 0
            mrc.header.origin.x = self.xi
            mrc.header.origin.y = self.yi
            mrc.header.origin.z = self.zi

            mrc.header.cella.x = self.xb * self.voxsp
            mrc.header.cella.y = self.yb * self.voxsp
            mrc.header.cella.z = self.zb * self.voxsp

            mrc.header.mapc = 1
            mrc.header.mapr = 2
            mrc.header.maps = 3

