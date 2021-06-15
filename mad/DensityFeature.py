import numpy as np
from copy import deepcopy
from .eqsp.eqsp import EQSP_Sphere

class DensityFeature(object):
    def __init__(self):
        # Detector level info
        self.voxel_val = 0
        self.oct_scale = -1
        self.coords = []
        self.map_coords = []
        self.subv_map_coords = []
        self.ratio = 0

        # Orientation assignment stuff
        self.eqsp_size = -1
        self.grad_box = []
        self.magn_box = []
        self.ar_magn = []
        self.ar_count = []
        self.norm_zone_counts = [] # to test stuff
        self.main_bin = -1
        self.sec_bin = -1
        self.list_bins = [] # list of bin_idx that are candidate for dominant orientations
        self.list_sec_bins = []
        self.to_dom_mat = []
        self.adj_sec_mat = []
        self.Rfinal = []

        # Descriptor stuff
        self.interp_grad_box = []
        self.ar_subcount = []
        self.lin_magn_sudo = []

    def set_detector_info(self, index, oct_scale, coords, map_coords, subv_map_coords, voxel_val):
        self.index = index
        self.oct_scale = oct_scale
        self.coords = coords
        self.map_coords = map_coords
        self.subv_map_coords = subv_map_coords
        self.voxel_val = voxel_val

    def set_orientator_info(self, eqsp_size, radius):
        self.eqsp_size       = eqsp_size
        self.box_size        = radius * 2 + 1
        self.box_side        = radius
        self.grad_box        = np.zeros( (self.box_size, self.box_size, self.box_size, 3) )
        self.magn_box        = np.zeros( (self.box_size, self.box_size, self.box_size, 1) )
        self.ar_magn         = np.zeros( eqsp_size )
        self.ar_count        = np.zeros( eqsp_size, dtype=np.int32)
        self.step_ar_counts  = []
        self.step_v_counts   = []

    def set_descriptor_info(self, subeqsp_size, radius):
        self.subeqsp_size    = subeqsp_size
        self.box_size        = radius * 2 + 1
        self.box_side        = radius

    def set_from_file_ori(self, index, main_bin, sec_bin, oct_scale, eqsp_size,
                                coord, map_coord, subv_map_coord, Rfinal, ar_count):
        self.eqsp_size = eqsp_size
        self.index = index
        self.main_bin = main_bin
        self.sec_bin = sec_bin
        self.oct_scale = oct_scale
        self.coords = coord
        self.map_coords = map_coord
        self.subv_map_coords = subv_map_coord
        self.Rfinal = Rfinal
        self.ar_count = ar_count

    def set_from_file_dsc(self, index, main_bin, sec_bin, oct_scale, eqsp_size, subeqsp_size,
                         coord, map_coord, subv_map_coord, Rfinal, descr):
        self.eqsp_size = eqsp_size
        self.subeqsp_size    = subeqsp_size
        self.index = index
        self.main_bin = main_bin
        self.sec_bin = sec_bin
        self.oct_scale = oct_scale
        self.coords = coord
        self.map_coords = map_coord
        self.subv_map_coords = subv_map_coord
        self.Rfinal = Rfinal
        self.lin_ar_subeqsp = descr

    #################
    # Visualization #
    #################

    def show(self):
        print("#############################")
        print("DF @o=%i: idx=%i main_bin=%i sec_bin=%i:"%(self.oct_scale, self.index, self.main_bin, self.sec_bin))
        print("Base: %i"%(self.eqsp_size))
        print("> Coords: %.3f %.3f %.3f"%(self.coords[0], self.coords[1], self.coords[2]))
        print("> Map coords: %.3f %.3f %.3f"%(self.map_coords[0], self.map_coords[1], self.map_coords[2]))
        print("> Subv coords:%.3f %.3f %.3f"%(self.subv_map_coords[0], self.subv_map_coords[1], self.subv_map_coords[2]))
        print("#############################")

    def show_occupancy(self):
        eqsp = EQSP_Sphere(size=self.eqsp_size)

        for belt_idx in range(len(eqsp.belt_l)):
            if not eqsp.max_belt_l%2:
                space = 3*(eqsp.max_belt_l - len(eqsp.belt_l[belt_idx])) // 2  + int(len(eqsp.belt_l[belt_idx])%2)*2
            if eqsp.max_belt_l%2:
                space = 3*(eqsp.max_belt_l - len(eqsp.belt_l[belt_idx])) // 2  + int(len(eqsp.belt_l[belt_idx])%2)
            str_vec = "" + " "*space
            for val in self.ar_count[eqsp.belt_l[belt_idx]]:
                str_vec += "%2i "%val
            print(str_vec)
        print()

    def show_suboccupancy(self):
        for idx in range(0,len(self.ar_suboccupancy),2):
            su1 = self.ar_suboccupancy[idx]
            su2 = self.ar_suboccupancy[idx+1]
            s =     "       %2i               %2i\n"%(su1[0], su2[0])
            s = s + " %2i %2i %2i %2i %2i   %2i %2i %2i %2i %2i\n"%(su1[1], su1[2], su1[3], su1[4], su1[5], su2[1], su2[2], su2[3], su2[4], su2[5])
            s = s + " %2i %2i %2i %2i %2i   %2i %2i %2i %2i %2i\n"%(su1[6], su1[7], su1[8], su1[9], su1[10], su2[6], su2[7], su2[8], su2[9], su2[10])
            s = s + "       %2i               %2i\n"%(su1[11], su2[11])
            print(s)
            print()

    def write_tcl_raw(self, step):

        # Writes a TCL script to run in VMD
        # > Displays feature around origin
        with open("Feature_raw_%i_step%i.tcl"%(self.index, step), "w") as f:
            f.write("proc vmd_draw_arrow {mol start end} {\n")
            f.write("   set middle [vecadd $start [vecscale 0.9 [vecsub $end $start]]]\n")
            f.write("   graphics $mol cylinder $start $middle radius 0.02\n")
            f.write("   graphics $mol cone $middle $end radius 0.10\n")
            f.write("}\n")

            f.write("mol new def.pdb\n")
            f.write("draw color red\n")
            for i in range(self.box_size):
                for j in range(self.box_size):
                    for k in range(self.box_size):
                        if self.magn_box[i,j,k] == 0 or (i==self.box_side and j==self.box_side and k==self.box_side ):
                            continue
                        f.write("draw arrow {%i %i %i} {%f %f %f}\n"%(0,0,0, self.grad_box[i,j,k,0], self.grad_box[i,j,k,1], self.grad_box[i,j,k,2]))
