# Class that loads information from EQ Sphere Partitions (Leopardi, Electronic Transactions on Numercial Analysis, 2006)
# > Using the author's Matlab's script, files containing informations about area boundaris (polar) and center
#   are written and then loaded here

# Sylvain Traeger / LBM - EPFL
# Last update: 12 June 2017
import numpy as np
from math import sin, cos

from sklearn.neighbors import NearestNeighbors

class EQSP_Sphere(object):
    def __init__(self, size=112):
        # Load area boundaries for feature sphere
        self.sphere_eqsp = []
        with open("mad/eqsp/sphere_%i.txt"%size, "r") as f:
            for line in f:
                # Order in file: min_theta, min_phi, max_theta, max_phi
                self.sphere_eqsp.append([float(x) for x in line.split(' ')])
        self.sphere_eqsp = np.array(self.sphere_eqsp)
        self.size = size

        # lead centers for each area of feature sphere in polar and cartesian coord
        self.p_centers_eqsp = []
        self.c_centers_eqsp = []
        with open("mad/eqsp/centers_%i.txt"%size, "r") as f:
            for line in f:
                # Order in file: theta, phi
                pol = [float(x) for x in line.split(' ')]
                self.p_centers_eqsp.append(pol)
                self.c_centers_eqsp.append(np.array([sin(pol[1])*cos(pol[0]),
                                                     sin(pol[1])*sin(pol[0]),
                                                     cos(pol[1])]))
        self.p_centers_eqsp = np.array(self.p_centers_eqsp)
        self.c_centers_eqsp = np.array(self.c_centers_eqsp)

        # Divide area indices according to the belt they belong to
        belt_l = []
        prev_area_val = -1
        belt_idx = -1
        max_belt_l = 0
        for idx in range(size):
            if self.area(idx)[1] != prev_area_val:
                belt_l.append([])
                belt_idx += 1
                belt_l[belt_idx].append(idx)
                prev_area_val = self.area(idx)[1]
            else:
                belt_l[belt_idx].append(idx)

        area_counter = 0
        for belt in belt_l:
            if len(belt) > max_belt_l:
                max_belt_l = len(belt)
                self.equator_idx = area_counter # To get polar bounds in Dfeature's wiggle_wiggle()
            area_counter += len(belt)

        self.belt_l = belt_l
        self.max_belt_l = max_belt_l

        # Average distance between area centers * 0.1 = threshold to readjust gradient decomposition
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(self.c_centers_eqsp)
        distances, indices = nbrs.kneighbors(self.c_centers_eqsp)
        self.feature_dist_thresh = np.average(distances[:,1])*0.1

    def p_center(self, idx):
        return self.p_centers_eqsp[idx]

    def c_center(self, idx):
        return self.c_centers_eqsp[idx]

    def area(self, idx):
        return self.sphere_eqsp[idx]

    def dist_thresh(self):
        return self.feature_dist_thresh

    def belt_indices(self, idx):
        return self.belt_l[idx]

    def belt_of_idx(self, idx):
        for i, b in zip(range(len(self.belt_l)), self.belt_l):
            if idx in b:
                return i

    def belt_equator_bounds(self):
        return self.sphere_eqsp[self.equator_idx]

