import time
import numpy as np
import warnings
from copy import deepcopy

from .eqsp.eqsp import EQSP_Sphere
from .math_utils import unit_vector, euler_rod_mat
from .DensityFeature import DensityFeature

PI = np.pi

class Orientator(object):
    def __init__(self, eqsp_size=112, main_ori=6, sec_ori=6, ori_radius=16, gw_sig=0, magn_weighted=False):
        # Build EQSPs
        self.eqsp_size    = eqsp_size
        self.eqsp         = EQSP_Sphere(size=self.eqsp_size)

        # Orientation parameters
        self.lim_main_ori = main_ori
        self.lim_sec_ori  = sec_ori

        # Cut-off for magnitudes when normalizing gradient. Below is considered 0
        self.cutoff_magn = 1e-5

        # Radius around anchor (diameter is hence radius + anchor + radius = 2*radius + 1, or 2* radius for descriptor after interpolation)
        if ori_radius %2:
            print("MaD> ERROR: radius %i invalid, must be even. Setting %i instead."%(ori_radius, ori_radius-1))
            ori_radius = ori_radius -1
        self.ori_radius = ori_radius//2

        # Sigma for gaussian window on orientation assignment
        self.gw_sig = gw_sig
        self.magn_weighted = magn_weighted

        # Mask for elements on sphere centered on keypoint => avoid bias from box corners
        # Coords for kernel on grid during orientation assignement
        dr = self.ori_radius
        coord_grid = np.mgrid[-dr:dr+1, -dr:dr+1, -dr:dr+1]
        sum_sq = np.sum(np.multiply(coord_grid, coord_grid), 0)

        # Sphere mask for orientation assignemnt: corners on cube are sources of bias (rotated cubes will have corners non matching)
        # > this mask puts a weight of 0 on these corners to improve orientation assignment
        # > for descriptor generation, we suppose we have a matching orientation so we just rely on gaussian window.
        self.sphere_mask_ori_dict = np.sqrt(sum_sq)
        self.sphere_mask_ori_dict[self.sphere_mask_ori_dict <= dr*1.05] = 1
        self.sphere_mask_ori_dict[self.sphere_mask_ori_dict > dr*1.05] = 0
        self.sphere_mask_ori_dict = self.sphere_mask_ori_dict.astype(int)

        # Gaussian weights
        if gw_sig:
            self.gauss_weight_ori_dict = np.exp(-1 * np.divide(sum_sq, 2*(gw_sig)**2))
        else:
            self.gauss_weight_ori_dict = np.ones_like(sum_sq)
        self.gauss_weight_ori_dict = np.multiply(self.gauss_weight_ori_dict, self.sphere_mask_ori_dict)

        # Timing steps
        self.time1 = 0
        self.time2 = 0
        self.time3 = 0
        self.time4 = 0
        self.time5 = 0


    #######
    # META
    #######

    def assign_orientations(self, ms, df_list):
        '''
        Assign orientation for every df in list.
        > Count gradient vectors according to the EQSP area they're directed at
        > Main orientations are defined by the area containing up to 80% of max count
        > Secondary orientations are defined similarly, but excluding pole.
        > In both cases, ambigous orientations (between 80% and 100% of max count) lead to duplication of df
        '''
        print("MaD> Orienting %i anchors..."%(len(df_list)))

        oriented_df_list = []
        # for idx in tqdm(range(len(df_list))):
        for idx in range(len(df_list)):
            df = df_list[idx]
            df.set_orientator_info(self.eqsp_size, self.ori_radius)
            if not self.step01_build_grad_nhood(ms.grad_list[df.oct_scale], df):
                continue

            if not self.step02_check_first(df):
                continue

            # DF is duplicated for every ambigous orientation
            for cand_bin in df.list_bins:
                cur_df = deepcopy(df)
                cur_df.main_bin = cand_bin
                is_good_first = self.step03_orient_first(cur_df)
                is_good_second = self.step04_check_second(cur_df)

                if not is_good_first or not is_good_second:
                    del cur_df
                    continue
                else:
                    for cand_sec_bin in cur_df.list_sec_bins:
                        cur_sec_df = deepcopy(cur_df)
                        cur_sec_df.sec_bin = cand_sec_bin

                        if self.step05_orient_second(cur_sec_df):
                            cur_sec_df.Rfinal = np.dot(cur_sec_df.adj_sec_mat, cur_sec_df.to_dom_mat)
                            oriented_df_list.append(cur_sec_df)
                        else:
                            del cur_sec_df

        return oriented_df_list

    #########
    # Steps #
    #########

    def step01_build_grad_nhood(self, grad, df):
        '''
        Gets the gradient patch around key point, gets magnitude and initial count/zone in EQSP sphere
        '''
        t1 = time.time()

        # Extract gradient around keypoint
        dx, dy, dz = df.coords
        sx, sy, sz, sg = grad.shape

        # Reject keypoints too close to map borders based on descriptor dimension
        # > Shouldn't happen if you set padding accordingly in Map_Space object
        if df.oct_scale == 1:
            xp, yp, zp = np.add(df.coords, df.box_side+1)
            xm, ym, zm = np.subtract(df.coords, df.box_side)
            if xm < 0      or ym < 0      or zm < 0  or \
               xp > sx - 1 or yp > sy - 1 or zp > sz - 1:
                self.step1_reject += 1
                self.time1 += time.time() - t1
                return False

            # Get gradient and magnitude
            df.grad_box = deepcopy(grad[xm:xp, ym:yp, zm:zp, :])
            df.magn_box = np.sqrt(np.sum(np.square(df.grad_box), -1))

            # Normalize gradient, avoiding where magn is 0
            maskx, masky, maskz = np.where(df.magn_box > self.cutoff_magn)
            df.grad_box[maskx, masky, maskz, :] = df.grad_box[maskx, masky, maskz,:] / np.expand_dims(df.magn_box[maskx, masky, maskz], -1)

            # Make mask by removing corners and apply gaussian window if applicable
            df.weight_mask = self.gauss_weight_ori_dict.copy()
            df.weight_mask[np.where(df.magn_box < self.cutoff_magn)] = 0
        else:
            xp, yp, zp = np.add(df.coords, df.box_side*2+1)
            xm, ym, zm = np.subtract(df.coords, df.box_side*2)
            if xm < 0      or ym < 0      or zm < 0  or \
               xp > sx - 1 or yp > sy - 1 or zp > sz - 1:
                self.step1_reject += 1
                self.time1 += time.time() - t1
                return False

            # Get gradient and magnitude
            df.grad_box = deepcopy(grad[xm:xp:2, ym:yp:2, zm:zp:2, :])
            df.magn_box = np.sqrt(np.sum(np.square(df.grad_box), -1))

            # Normalize gradient, avoiding where magn is 0
            maskx, masky, maskz = np.where(df.magn_box > self.cutoff_magn)
            df.grad_box[maskx, masky, maskz, :] = df.grad_box[maskx, masky, maskz,:] / np.expand_dims(df.magn_box[maskx, masky, maskz], -1)

            # Make mask by removing corners and apply gaussian window if applicable
            df.weight_mask = self.gauss_weight_ori_dict.copy()
            df.weight_mask[np.where(df.magn_box < self.cutoff_magn)] = 0
        self.time1 += time.time() - t1
        return True

    def step02_check_first(self, df):
        '''
        Count gradient vectors based on their direction, following bins dictated by EQSP sphere
        > Bins with largest counts are considered as main bins
        '''
        t2 = time.time()

        self.process_df_gradient(df)

        # Assessment of anchor viability: count candidate bins and check they're not too many
        df.list_bins = np.where(df.ar_count>max(df.ar_count)*0.8)[0]
        if len(df.list_bins) > (self.lim_main_ori):
            self.time2 += time.time() - t2
            return False

        self.time2 += time.time() - t2
        return True

    def step03_orient_first(self, df):
        '''
        Move DF so that its main bin is at the pole
        > First step for rotation invariance
        > update arrays accordingly
        '''
        t3 = time.time()

        # Check if main bin isn't already at the pole
        if df.main_bin != 0:
            # Get cartesian coordinates of points of interests, then get angle + axis of rotation
            input_bin_c_center = unit_vector(self.eqsp.c_center(df.main_bin))
            angle = np.arccos(np.clip(np.dot(input_bin_c_center, [0,0,1]), -1.0, 1.0))
            ax_rot = unit_vector(np.cross(input_bin_c_center, [0,0,1]))

            res = self.rotate_bins_to_dom(df, angle, ax_rot)

            self.process_df_gradient(df)

            self.time3 += time.time() - t3
            return res
        else:
            df.to_dom_mat = np.identity(3)
            self.time3 += time.time() - t3
            return True

    def step04_check_second(self, df):
        '''
        Count gradient vectors based on their direction, following bins dictated by EQSP sphere
        > Bins with largest counts except poles are considered as secondary bins
        > secondary bins will be aligned to theta=0 to have rotation invariance
        '''
        t4 = time.time()

        # Ignore both poles, and first belt to ignore "bleeding" of pole into first belt, leading to bias
        # pos = 1 + len(self.eqsp.belt_l[1])
        # just ignore pole
        pos = 1

        not_pole = df.ar_count[pos:-1].copy()
        if np.amax(not_pole) == 0:
            self.time4 += time.time() - t4
            return False
        not_pole = np.array(not_pole / np.amax(not_pole) * 50, dtype=np.int32)

        # Assessment of anchor viability: count candidate bins and check they're not too many
        df.list_sec_bins = np.where(not_pole>max(not_pole)*0.8)[0] + pos

        if len(df.list_sec_bins) > self.lim_sec_ori:
            self.time4 += time.time() - t4
            return False

        self.time4 += time.time() - t4
        return True

    def step05_orient_second(self, df):
        '''
        Move DF so that its secondary bin is at theta=0, making it fully rotation invariant
        > update arrays accordingly
        '''
        t5 = time.time()
        # Rotate around the Z axis
        # > In the first belt, check which theta angle gives the best first bin
        # > Rotate bins to that angle
        ax_rot = [0,0,1]

        # Rotate so that center of secondary bin is at theta=0 (BAK)
        # ftheta = -1 * self.eqsp.area(df.sec_bin)[0]

        # Rotate so that center of secondary bin is at position of first of the belt
        theta_belt_first = self.eqsp.p_center(self.eqsp.belt_l[self.eqsp.belt_of_idx(df.sec_bin)][0])[0]
        ftheta = -1 * (self.eqsp.p_center(df.sec_bin)[0] - theta_belt_first)

        # Rotate bins
        res = self.rotate_bins_to_dom(df, ftheta, ax_rot, sec=True)
        self.time5 += time.time() - t5

        if 0:
            #### THIS IS TO GET ARRAY OF ZONES FOR PLOTTING
            self.process_df_gradient(df)

        return res

    ##########
    # Others #
    ##########
    def show_timing(self):
        times = [
                 self.time1,
                 self.time2,
                 self.time3,
                 self.time4,
                 self.time5
                ]
        print("MaD> Step timing:")
        s = "          "
        for i, t in enumerate(times):
            s = s + "%.1f  "%(t)
        s = s + " Total: %.2f s"%(np.sum(times))
        print(s)

    def rotate_bins_to_dom(self, df, angle, ax_rot, sec=False):
        # Rotate gradient as to have the dominant orientation in the EQSP northern pole or first bin of first belt.
        rot_mat = euler_rod_mat(ax_rot, angle)
        if sec:
            df.adj_sec_mat = np.array(rot_mat)
        else:
            df.to_dom_mat = np.array(rot_mat)
        df.grad_box = self.rotate_gradient(rot_mat, df)

        return True

    def rotate_gradient(self, rot_mat, df):
        # Rotate grad_box of DF 'df' with rot_mat.
        return np.matmul(df.grad_box, rot_mat.T)

    def process_df_gradient(self, df):
        # Get gradient vectors in spherical coords for easy binning
        th_ar = np.arctan2(df.grad_box[:,:,:,1], df.grad_box[:,:,:,0])

        # Into 0-2pi range
        th_ar[th_ar < 0] += 2 * PI # adjust for EQSP range

        # For zones that exceed 2pi range
        sth_ar = th_ar + 2*PI

        # Polar angle phi
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                ph_ar = np.arccos(df.grad_box[:,:,:,2])
            except Warning:
                ph_ar = np.arccos(np.clip(df.grad_box[:,:,:,2], -1, 1))

        # Count per zone
        for area_idx in range(self.eqsp.size):
            area = self.eqsp.area(area_idx) # polar angle ranges of that area

            # Get info from current EQSP area
            th_mask = np.logical_and(th_ar < area[2], th_ar > area[0])
            sth_mask = np.logical_and(sth_ar < area[2], sth_ar > area[0])
            ph_mask = np.logical_and(ph_ar < area[3], ph_ar > area[1])
            area_mask = np.logical_and( np.logical_or(th_mask, sth_mask), ph_mask)

            # Count gradient vectors in zone
            df.ar_count[area_idx] = np.sum(df.weight_mask[area_mask])

        if not np.amax(df.ar_count):
            return False

        df.step_v_counts.append(df.ar_count)
        df.ar_count = np.array(df.ar_count / np.amax(df.ar_count) * 50, dtype=np.int32)

        # To test steps
        df.step_ar_counts.append(df.ar_count.copy())

    def check_nb_orientations(self, df):
        nb_orient = np.count_nonzero(df.ar_count > 0.8 * 50)
        if nb_orient > self.lim_main_ori:
            return False
        if nb_orient == 0:
            return False

        return True

    def check_nb_sec_orientations(self, df):
        not_pole = df.ar_count[1:-1].copy()
        not_pole = np.array(not_pole / np.amax(not_pole) * 50, dtype=np.int32)
        nb_orient = np.count_nonzero(not_pole > 0.8 * 50)
        if nb_orient > self.lim_sec_ori:
            return False
        if nb_orient == 0:
            return False

        return True

    def is_bin_dominant(self, df):
        # Check if pole still dominant after rotation (80% of max)
        if df.ar_count[0] < 0.8 * 50:
            return False
        return True

    def polar_to_cart(self, theta, phi):
        return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])


    #######
    # I/O #
    #######

    def write_df_to_npy(self, df_list, outname):
        df_ar = [np.concatenate([[df.index, df.main_bin, df.sec_bin, df.oct_scale, df.eqsp_size], \
                 df.coords, df.map_coords, df.subv_map_coords, df.Rfinal.flatten(), df.ar_count]) for df in df_list]
        df_ar = np.array(df_ar)
        np.save(outname, df_ar, allow_pickle=False)

    def load_df_from_npy(self, ori_file):
        df_list = []
        df_coords = []
        index_list = []
        data_df = np.load(ori_file)
        for data in data_df:
            index, main_bin, sec_bin, oct_scale, eqsp_size = data[:5].astype(int)

            if eqsp_size != self.eqsp_size:
                print("Descript> Aborting loading; EQSP size does not match")
                return []

            coords = np.array(data[5:8])
            map_coords = np.array(data[8:11])
            subv_map_coords = np.array(data[11:14])
            R = np.array(data[14:23]).reshape((3,3))
            ar_count = np.array(data[23:])
            df = DensityFeature()

            df.set_from_file_ori(index, main_bin, sec_bin, oct_scale, eqsp_size,
                                 coords, map_coords, subv_map_coords, R, ar_count)
            df_list.append(df)
            if index not in index_list:
                index_list.append(index)
                df_coords.append(subv_map_coords)

        print("MaD> Loaded %i descriptors."%len(df_list))
        return df_list, np.array(df_coords)
