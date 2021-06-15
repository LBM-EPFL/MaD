import sys
import time
import numpy as np
import warnings

import h5py

from .eqsp.eqsp import EQSP_Sphere
from .DensityFeature import DensityFeature

PI = np.pi

class Descriptor(object):
    def __init__(self, subeqsp_size=16, dsc_radius=16, dsc_size=64):
        # EQSPs
        self.subeqsp_size = subeqsp_size
        self.subeqsp = EQSP_Sphere(size=self.subeqsp_size)

        # Cut-off for magnitudes when normalizing gradient. Below is considered 0
        self.cutoff_magn = 1e-12

        # Radius around anchor (diameter is hence radius + anchor + radius = 2*radius + 1, or 2* radius for descriptor after interpolation)
        if dsc_radius %2:
            print("MaD>> ERROR: radius %i invalid, must be even. Setting %i instead."%(dsc_radius, dsc_radius-1))
            dsc_radius = dsc_radius -1
        self.dsc_radius = dsc_radius//2

        # Number of sub.regions
        self.dsc_size = dsc_size

        # Interpolation support for upsampled grid (df.oct_scale = 0), or base grid (df.oct_scale = 1)
        dr = self.dsc_radius
        self.dsc_layout = {}
        self.dsc_layout[0] = np.rollaxis(np.array(np.mgrid[-2*dr+1:2*dr+1:2, -2*dr+1:2*dr+1:2, -2*dr+1:2*dr+1:2]), 0, 4)
        self.dsc_layout[1] = np.rollaxis(np.array(np.mgrid[-dr+0.5:dr+0.5, -dr+0.5:dr+0.5, -dr+0.5:dr+0.5]), 0, 4)

        # Slices to divide patch into descriptor subregions
        if self.dsc_size == 64:
            dr = self.dsc_radius
            s1 = slice(0, dr//2)
            s2 = slice(dr//2, dr)
            s3 = slice(dr, 3*dr//2)
            s4 = slice(3*dr//2, 2*dr)
            self.sub_slices = [
                               (s1, s1, s1), (s1, s1, s2), (s1, s1, s3), (s1, s1, s4),
                               (s2, s1, s1), (s2, s1, s2), (s2, s1, s3), (s2, s1, s4),
                               (s3, s1, s1), (s3, s1, s2), (s3, s1, s3), (s3, s1, s4),
                               (s4, s1, s1), (s4, s1, s2), (s4, s1, s3), (s4, s1, s4),

                               (s1, s2, s1), (s1, s2, s2), (s1, s2, s3), (s1, s2, s4),
                               (s2, s2, s1), (s2, s2, s2), (s2, s2, s3), (s2, s2, s4),
                               (s3, s2, s1), (s3, s2, s2), (s3, s2, s3), (s3, s2, s4),
                               (s4, s2, s1), (s4, s2, s2), (s4, s2, s3), (s4, s2, s4),

                               (s1, s3, s1), (s1, s3, s2), (s1, s3, s3), (s1, s3, s4),
                               (s2, s3, s1), (s2, s3, s2), (s2, s3, s3), (s2, s3, s4),
                               (s3, s3, s1), (s3, s3, s2), (s3, s3, s3), (s3, s3, s4),
                               (s4, s3, s1), (s4, s3, s2), (s4, s3, s3), (s4, s3, s4),

                               (s1, s4, s1), (s1, s4, s2), (s1, s4, s3), (s1, s4, s4),
                               (s2, s4, s1), (s2, s4, s2), (s2, s4, s3), (s2, s4, s4),
                               (s3, s4, s1), (s3, s4, s2), (s3, s4, s3), (s3, s4, s4),
                               (s4, s4, s1), (s4, s4, s2), (s4, s4, s3), (s4, s4, s4),
                              ]
        elif self.dsc_size == 27:
            fl = 2*dr
            s1 = slice(0, fl//3)
            s2 = slice(fl//3, 2*fl//3)
            s3 = slice(2*fl//3, fl)
            self.sub_slices = [
                (s1, s1, s1), (s1, s1, s2), (s1, s1, s3),
                (s2, s1, s1), (s2, s1, s2), (s2, s1, s3),
                (s3, s1, s1), (s3, s1, s2), (s3, s1, s3),

                (s1, s2, s1), (s1, s2, s2), (s1, s2, s3),
                (s2, s2, s1), (s2, s2, s2), (s2, s2, s3),
                (s3, s2, s1), (s3, s2, s2), (s3, s2, s3),

                (s1, s3, s1), (s1, s3, s2), (s1, s3, s3),
                (s2, s3, s1), (s2, s3, s2), (s2, s3, s3),
                (s3, s3, s1), (s3, s3, s2), (s3, s3, s3),
                                 ]
        elif self.dsc_size == 8:
            s1 = slice(0, dr)
            s2 = slice(dr, 2*dr)
            self.sub_slices = [
                (s1, s1, s2), (s1, s1, s1), (s1, s2, s2), (s1, s2, s1),
                (s2, s1, s2), (s2, s1, s1), (s2, s2, s2), (s2, s2, s1)
                                ]
        elif self.dsc_size == 1:
            fl = 2*dr
            s1 = slice(0, fl)
            self.sub_slices = [(s1, s1, s1)]
        else:
            print("MaD>> ERROR: invalid dsc size %i"%self.dsc_size)
            sys.exit(1)

        # Timing steps
        self.time6 = 0


    #######
    # META
    #######

    def generate_descriptors(self, ms, df_list):
        '''
        For every DF, interpolate gradient from map_space (more accurate than interpolating just on patch)
        Then, divide DF space into 8 quadrants, where gradient is sorted in a number of bins as defined by subeqsp_size
        Final descriptors are hence NxNxNxSubEQSPzones, where N is from (boxsize_range) depennding on octave of DF
        '''
        print("MaD> Generating descriptors from %i oriented anchors..."%len(df_list))
        for idx, df in enumerate(df_list):
            df.set_descriptor_info(self.subeqsp_size, self.dsc_radius)
            self.step06_distribute_subeqsp(ms, df)
        return df_list


    #########
    # Steps #
    #########

    def step06_distribute_subeqsp(self, ms, df):
        '''
        Interpolate + distribute gradient magnitudes to build descriptor
        > Divided into 8 quadrants with each a number of zone equal to subeqsp_size
        > Achieves robustness to noise, displacement, conformational changes
        '''
        t6 = time.time()

        # Get target coords for interpolated grid points (NOT gradient vectors)
        dsc_layout_inv = np.matmul(self.dsc_layout[df.oct_scale], np.linalg.inv(df.Rfinal).T)
        dsc_layout_inv = np.add(dsc_layout_inv, df.coords)

        # Get gradient + magnitudes from (unrotated) original map at interpolated coordinates
        # > These coordinates, after rotation to proper referential, match those of the final descriptor perfectly.
        # > This allows to use an interpolator based on the gradient of the whole map rather than just on the patch
        # > And to not have to rotate / resimulate maps for each descriptor
        # > And this avoids border effects
        try:
            df.interp_grad_box = ms.rgi_space[df.oct_scale](dsc_layout_inv)
        except:
            df.lin_ar_subeqsp = np.zeros((len(self.sub_slices) * self.subeqsp_size)).astype(np.int16)
            # print("MaD>> ERROR: Bad interpolation RGI out of bounds; setting DSC to 0 vect of shape %s, dtype %s\n"%(str(df.lin_ar_subeqsp.shape), str(df.lin_ar_subeqsp.dtype)))
            # print(df.map_coords, df.oct_scale, ms.gauss_list[df.oct_scale].shape)
            # print(ms.xi, ms.yi, ms.zi)
            # print(np.amin(dsc_layout_inv), np.amax(dsc_layout_inv))
            self.time6 += time.time() - t6
            return
        df.interp_magn_box = np.sqrt(np.sum(np.square(df.interp_grad_box), -1))

        # Normalize gradient and magnitudes, rotate to oriented referential
        ix, iy, iz = np.where(df.interp_magn_box > self.cutoff_magn)
        df.interp_grad_box[ix, iy, iz,:] = df.interp_grad_box[ix, iy, iz,:] / np.expand_dims(df.interp_magn_box[ix, iy, iz], -1)
        df.interp_grad_box = np.matmul(df.interp_grad_box, df.Rfinal.T)

        # Get gradient vectors in spherical coords for easy binning
        th_ar = np.arctan2(df.interp_grad_box[:,:,:,1], df.interp_grad_box[:,:,:,0])

        # Into 0-2pi range
        th_ar[th_ar < 0] += 2 * np.pi

        # For zones that exceed 2pi range
        sth_ar = th_ar + 2*np.pi

        # Polar angle phi
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                ph_ar = np.arccos(df.interp_grad_box[:,:,:,2])
            except Warning:
                ph_ar = np.arccos(np.clip(df.interp_grad_box[:,:,:,2], -1, 1))
        zone_assignments = np.zeros(df.interp_magn_box.shape, dtype=np.int16)

        # Count gradient vectors per EQ area
        for area_idx in range(self.subeqsp_size):
            # Get info from current EQSP area
            area = self.subeqsp.area(area_idx) # polar angle ranges of that area

            # Masks for that zone
            th_mask = np.logical_and(th_ar < area[2], th_ar > area[0])
            sth_mask = np.logical_and(sth_ar < area[2], sth_ar > area[0])
            ph_mask = np.logical_and(ph_ar < area[3], ph_ar > area[1])
            area_mask = np.logical_and( np.logical_or(th_mask, sth_mask), ph_mask)

            # Assign zone idx to voxels
            zone_assignments[area_mask] = area_idx

        # Remove low-density zones from being counted
        zone_assignments[df.interp_magn_box < 1e-5] = -1

        # Get counts per zone per descriptor region
        arbin_subeqsp = []
        for subeq in range(len(self.sub_slices)):
            arbin_subeqsp.append([np.count_nonzero(zone_assignments[self.sub_slices[subeq]] == sub_idx) for sub_idx in range(self.subeqsp_size)])

        # Flatten for final descriptor
        df.lin_ar_subeqsp = np.array([item for sublist in arbin_subeqsp for item in sublist], dtype=np.int16)

        self.time6 += time.time() - t6

        return True

    ##########
    # Others #
    ##########

    def show_timing(self):
        times = [self.time6]
        print("MaD> Step timing:")
        s = "          "
        for i, t in enumerate(times):
            s = s + "%.1f  "%(t)
        s = s + " Total: %.2f s"%(np.sum(times))
        print(s)


    def polar_to_cart(self, theta, phi):
        return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])


    #######
    # I/O #
    #######

    def save_h5(self, df_list, outname):
        dsc_ar = np.array([df.lin_ar_subeqsp for df in df_list])
        info_ar = np.array([[df.index, df.main_bin, df.sec_bin, df.oct_scale, df.eqsp_size, df.subeqsp_size] for df in df_list]).astype(np.uint16)
        coords_ar = [[df.coords, df.map_coords, df.subv_map_coords] for df in df_list]
        rot_ar = np.array([df.Rfinal for df in df_list])

        hf = h5py.File(outname, 'w')
        hf.create_dataset('dsc', data=dsc_ar)
        hf.create_dataset('info', data=info_ar)
        hf.create_dataset('coords', data=coords_ar)
        hf.create_dataset('rot', data=rot_ar)
        hf.close()

    def load_h5(self, input_name):
        hf = h5py.File(input_name, 'r')
        dsc = hf.get('dsc')
        coords = hf.get('coords')
        info = hf.get('info')
        rot = hf.get('rot')

        df_list = []
        for d, c, i, r in zip(dsc, coords, info, rot):
            df = DensityFeature()
            df.set_from_file_dsc(i[0], i[1], i[2], i[3], i[4], i[5],
                             c[0], c[1], c[2], r, d)
            df_list.append(df)
        hf.close()
        print("MaD> Loaded %i anchors."%len(df_list))
        return df_list

