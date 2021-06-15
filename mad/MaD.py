import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as mplot

from copy import deepcopy
from operator import itemgetter
from scipy.spatial import cKDTree
from itertools import combinations
from sklearn.utils.extmath import cartesian

from .MapSpace import MapSpace
from .Detector import Detector
from .Orientator import Orientator
from .Descriptor import Descriptor
from .DensityFeature import DensityFeature
from .PDB import PDB
from .Dmap import Dmap
from .structure_utils import move_copy_structure, refine_pdb, get_overlap
from .math_utils import get_rototrans_SVD
from .eqsp.eqsp import EQSP_Sphere


class MaD(object):

    def __init__(self):
        # Files given through add_() functions
        self.input_map = None
        self.input_subunits = {}
        self.input_ensembles = {}

        # Processed structures being described and docked
        self.processed_map = None
        self.processed_subunits = {} # key: related to filename. Elements: filename, n_copies
        self.processed_ensembles = {} # key: ensemble directory. Elements: dict for each frame, same model as processed_subunits

        # Dict of solutions for assembly building
        self.buildable_subunits = {}

        # Output stuff
        self.out_folder = None
        self.dsc_dict = {}


    def add_subunit(self, sub_filename_folder, n_copies=1, identifier=""):
        assert os.path.exists(sub_filename_folder), "MaD> subunit or ensemble not found: %s"%sub_filename_folder

        # Single structure
        if os.path.isfile(sub_filename_folder):
            filename = os.path.splitext(os.path.split(sub_filename_folder)[-1])[0]
            sub_key = identifier if identifier != "" else filename
            if sub_key in self.input_subunits.keys():
                print("MaD> subunit %s already added; overwriting"%filename)
            self.input_subunits[sub_key] = [sub_filename_folder, n_copies]
            print("MaD> Added: subunit %s"%sub_filename_folder)

        # Ensemble
        elif os.path.isdir(sub_filename_folder):
            folder_name = os.path.basename(os.path.normpath(sub_filename_folder))
            sub_key = identifier if identifier != "" else folder_name
            frame_list = [os.path.join(sub_filename_folder, x) for x in os.listdir(sub_filename_folder) if x.split(".")[-1] in ["pdb", "PDB"]]
            if not len(frame_list):
                print("MaD> No PDB files found in ensemble folder %s"%sub_filename_folder)
            else:
                self.input_ensembles[sub_key] = {}
                for i, frame in enumerate(frame_list):
                    frame_key = os.path.splitext(os.path.split(frame)[-1])[0]
                    print(frame_key)
                    self.input_ensembles[sub_key][frame_key] = [frame, n_copies]
                print("MaD> Added: ensemble %s of %i frames"%(sub_key, i))
        else:
            print("MaD> Error: %s not a valid structure or ensemble"%sub_filename_folder)

    def add_map(self, input_map, resolution, isovalue=0):
        assert os.path.exists(input_map), "MaD> Exp. map not found: %s"%input_map
        assert resolution > 0, "MaD> Map cannot have a negative resolution"
        map_name = os.path.splitext(os.path.split(input_map)[-1])[0]

        self.resolution = resolution
        self.isovalue = isovalue
        self.input_map = input_map
        self.map_name = map_name
        print("MaD> Added: density map %s, resolution %.2f A"%(self.map_name, self.resolution))


    def run(self, transform_subunits=False, detect_sigma=2.0, presmooth_sigma=1, ori_eqsp_size=112, dsc_eqsp_size=16, dsc_subregions=64, patch_size=16, cc_threshold=0.6, weight_threshold=4, n_samples=60):
        # Mostly useful for benchmarks from pre-fitted structures.
        # This flag activates the moving of subunits away from their position
        # So results are not biased and reflect real scenarios
        self.transform_subunits = transform_subunits

        # Check that enough data have been added and prepare folders
        self.check_preprocess_data()

        # Run
        self.get_descriptors(detect_sigma=detect_sigma, presmooth_sigma=presmooth_sigma, patch_size=patch_size,
                             ori_eqsp_size=ori_eqsp_size, dsc_eqsp_size=dsc_eqsp_size, dsc_subregions=dsc_subregions)
        self.get_solutions(cc_threshold=cc_threshold, weight_threshold=weight_threshold, n_samples=n_samples)


    def check_preprocess_data(self):
        # Check that enough files have been defined
        if self.input_map is None or not (len(self.input_subunits.keys()) + len(self.input_ensembles.keys())):
            print("MaD> Make sure you have defined at least one component and a density map")
            if self.input_map is not None:
                print("     > Map: %s"%self.input_map)
            for k in self.input_subunits.keys():
                print("     > Subunit: %s"%k)
            return

        # Setup folders
        self._prep_files_folders()


    def get_descriptors(self, detect_sigma=2.0, presmooth_sigma=1, ori_eqsp_size=112, dsc_eqsp_size=16, dsc_subregions=64, patch_size=16):
        # describe map
        dsc_outname = f"dsc_db/{self.map_name}_res{self.resolution}_iso{self.isovalue}_detSig{detect_sigma}_presmooth{presmooth_sigma}_patch{patch_size}_orieqsp{ori_eqsp_size}_dsceqsp{dsc_eqsp_size}_subregions{64}.h5"
        if not os.path.exists(dsc_outname):
            print("\nMaD> Processing map %s"%self.map_name)
            descriptors = self._describe_struct(self.processed_map, detect_sigma, presmooth_sigma, ori_eqsp_size, dsc_eqsp_size, dsc_subregions, patch_size)
            self._save_descriptors(descriptors, dsc_outname)
        else:
            descriptors = self._load_descriptors(dsc_outname)
            print("MaD> %i descriptors for %s found in database"%(len(descriptors), self.map_name))
        self.map_dsc = descriptors

        # describe subunits
        for k in self.processed_subunits.keys():
            dsc_outname = f"dsc_db/{k}_res{self.resolution}_iso{self.isovalue}_detSig{detect_sigma}_presmooth{presmooth_sigma}_patch{patch_size}_orieqsp{ori_eqsp_size}_dsceqsp{dsc_eqsp_size}_subregions{64}.h5"

            if not os.path.exists(dsc_outname):
                print("\nMaD> Processing subunit %s"%k)
                descriptors = self._describe_struct(self.processed_subunits[k][0], detect_sigma, presmooth_sigma, ori_eqsp_size, dsc_eqsp_size, dsc_subregions, patch_size)
                self._save_descriptors(descriptors, dsc_outname)
            else:
                descriptors = self._load_descriptors(dsc_outname)
                print("MaD> %i descriptors for %s found in database"%(len(descriptors), k))

            self.dsc_dict[k] = descriptors

        # describe ensembles
        for ek in self.processed_ensembles.keys():
            ensemble = self.processed_ensembles[ek]
            self.dsc_dict[ek] = {}
            print("\nMaD> Describing ensemble %s"%ek)
            for fk in ensemble.keys():
                dsc_outname = f"dsc_db/{fk}_res{self.resolution}_iso{self.isovalue}_detSig{detect_sigma}_presmooth{presmooth_sigma}_patch{patch_size}_orieqsp{ori_eqsp_size}_dsceqsp{dsc_eqsp_size}_subregions{64}.h5"

                if not os.path.exists(dsc_outname):
                    print("MaD> Describing %s-%s"%(ek, fk))
                    descriptors = self._describe_struct(ensemble[fk][0], detect_sigma, presmooth_sigma, ori_eqsp_size, dsc_eqsp_size, dsc_subregions, patch_size)
                    self._save_descriptors(descriptors, dsc_outname)
                else:
                    descriptors = self._load_descriptors(dsc_outname)
                    print("MaD> Loaded %i descriptors for ensemble %s, frame %s"%(len(descriptors), ek, fk))

                # For ensembles, I switched to just setting the saved descriptors. This enables larger ensembles to be processed at once
                # as all frames may not fit in memory.
                # descriptors are reloaded when matching
                # self.dsc_dict[fk] = descriptors
                self.dsc_dict[fk] = dsc_outname


    def get_solutions(self, cc_threshold=0.6, weight_threshold=4, n_samples=120):
        # Dock individual subunits: match dsc, filter pairs, and refine solutions
        for k in self.processed_subunits.keys():
            # Adjust the amount of descriptor pairs to consider according to the expected number of copies of that subunit
            pdbfile, n_copies = self.processed_subunits[k]

            # Match subunit dsc with map's dsc, get solutions.
            solutions_filename_list = self._match_filter_refine(pdbfile, n_copies, k, cc_threshold, weight_threshold, n_samples)
            if len(solutions_filename_list):
                self.buildable_subunits[k] = [n_copies, solutions_filename_list]

        # Dock individual frames of ensembles
        # > merge solutions of all frames by ensemble for easier assembly building afterwards
        for ek in self.processed_ensembles.keys():
            ensemble = self.processed_ensembles[ek]
            n_copies = ensemble[list(ensemble.keys())[0]][1]
            self.buildable_subunits[ek] = [n_copies, []]
            for fk in ensemble.keys():
                # Adjust the amount of descriptor pairs to consider according to the expected number of copies of that subunit
                pdbfile, n_copies = ensemble[fk]

                # Match subunit dsc with map's dsc, get solutions.
                solutions_filename_list = self._match_filter_refine(pdbfile, n_copies, fk, cc_threshold, weight_threshold, n_samples)
                if len(solutions_filename_list):
                    [self.buildable_subunits[ek][1].append(sol) for sol in solutions_filename_list]


    def build_assembly(self, max_models=10, max_overlap_complex=0.1):
        # Allow assemblies / subcomplexes with a maximum of 10% overlap by default
        self.max_overlap_complex = max_overlap_complex

        # Max number of models to generate. Usually less than the default values are generated, and if more are generated, the best are among the first 10.
        self.max_models = max_models

        # Check that we have solutions
        if not len(self.buildable_subunits.keys()):
            print("MaD> No solutions found. Please run() first or adjust parameters if you did not get any solution.")
            return

        # No assembly to build if we have a monomer
        if sum([self.buildable_subunits[k][0] for k in self.buildable_subunits.keys()]) == 1:
            print("MaD> No assembly to build from a monomeric structure")
            return

        # Build complex from the single component
        if len(self.buildable_subunits.keys()) == 1:
            sub_key = list(self.buildable_subunits.keys())[0]
            self._build_from_single(sub_key, homomultimer=True)

        # Or build subcomplexes if a component is present in > 1 copies
        else:
            sub_sol_dict = {}
            for sub_key in self.buildable_subunits.keys():
                subcomplexes = self._build_from_single(sub_key)
                sub_sol_dict[sub_key] = subcomplexes

            # Build final assembly models
            self._build_models(sub_sol_dict)


    def score_ensembles(self):
        # Ignore if no ensemble was registered
        if len(self.processed_ensembles.keys()) == 0:
            print("MaD> No ensembles were provided and/or processed")
            return

        for ek in self.processed_ensembles.keys():
            ensemble = self.processed_ensembles[ek]
            csv_key_filename_list = [[fk, os.path.join(self.out_folder, "Solutions_refined_%s.csv"%fk)] for fk in sorted(ensemble.keys())]
            df_list = []
            for key_csv in csv_key_filename_list:
                key, csv = key_csv
                df = pd.read_csv(csv)
                df["StructID"] = [key] * df.shape[0]
                df_list.append(df)
            all_sols = pd.concat(df_list)
            all_sols.sort_values(by="mCC", ascending=False)

            ranking = []
            rep_list = []
            wgt_list = []
            mCC_list = []
            rwc_list = []
            for k in np.unique(sorted(ensemble.keys())):
                mean_rep = all_sols[all_sols["StructID"] == k]["Repeatability"].mean()
                mean_wgt = all_sols[all_sols["StructID"] == k]["Weight"].mean()
                mean_mCC = all_sols[all_sols["StructID"] == k]["mCC"].mean()
                mean_rwc = all_sols[all_sols["StructID"] == k]["RWmCC"].mean()
                ranking.append([k, mean_rep, mean_wgt, mean_mCC, mean_rwc])
                rep_list.append(mean_rep)
                wgt_list.append(mean_wgt)
                mCC_list.append(mean_mCC)
                rwc_list.append(mean_rwc)
            rep_ranking = sorted(ranking, key=itemgetter(1), reverse=True)
            wgt_ranking = sorted(ranking, key=itemgetter(2), reverse=True)
            mcc_ranking = sorted(ranking, key=itemgetter(3), reverse=True)
            rwc_ranking = sorted(ranking, key=itemgetter(4), reverse=True)

            print("MaD> Ranking for ensemble %s: "%ek)
            print("     Top 3 - Repeatability:")
            for i in range(3):
                print("     %i: %6.2f %s"%(i+1, rep_ranking[i][1], rep_ranking[i][0]))
            print("\n     Top 3 - Weight:")
            for i in range(3):
                print("     %i: %6.2f %s"%(i+1, wgt_ranking[i][2], wgt_ranking[i][0]))
            print("\n     Top 3 - Cross-corr.:")
            for i in range(3):
                print("     %i: %6.2f %s"%(i+1, mcc_ranking[i][3], mcc_ranking[i][0]))
            print("\n     Top 3 - MaD score:")
            for i in range(3):
                print("     %i: %6.2f %s"%(i+1, rwc_ranking[i][4], rwc_ranking[i][0]))

            fig, axes = mplot.subplots(nrows=1, ncols=4, figsize=(12,5))
            n_bars = len(rep_list) + 1
            for ax, l, n in zip(axes, [rep_list, wgt_list, mCC_list, rwc_list], ["Avg. R", "Avg. |clust|", "Avg. CC", "Avg. S"]):
                ax.bar(range(1, n_bars), l)
                ax.set_xticks(range(1, n_bars))
                ax.set_xticklabels(["C%i"%i for i in range(1, n_bars)], rotation=90)
                ax.set_title(n)

            mplot.tight_layout()
            mplot.savefig(os.path.join(self.out_folder, "Plot_score_ensemble.png"), dpi=600)


##################################
##################################

    def _prep_files_folders(self):
        # Create results folders if not present
        if not os.path.exists("results"):
            os.mkdir("results")
        if not os.path.exists("dsc_db"):
            os.mkdir("dsc_db")

        # Make results folder
        sub_keys_list = ["%sx%i"%(k, self.input_subunits[k][1]) for k in sorted(self.input_subunits.keys())]
        ens_keys_list = ["%sx%i"%(k, self.input_ensembles[k][list(self.input_ensembles[k].keys())[0]][1]) for k in sorted(self.input_ensembles.keys())]
        comp_str = ".".join([key for key in sub_keys_list + ens_keys_list])
        out_folder = f"results/{self.map_name}_{comp_str}_res{self.resolution:.3f}_iso{self.isovalue:.3f}"
        if os.path.exists(out_folder):
            idx = 1
            out_folder = out_folder + "_%i"
            while os.path.exists(out_folder%idx):
                idx += 1
            out_folder = out_folder%idx
        os.mkdir(out_folder)
        self.out_folder = out_folder
        print("MaD> Created output folder: %s"%out_folder)

        # Subfolder with pre-processed structures
        init_path = os.path.join(out_folder, "initial_files")
        os.mkdir(init_path)

        # Pre-process structures

        # > density map
        # Load map, apply isovalue, remove void, and save to folder
        if os.path.splitext(self.input_map)[-1].lower() in [".sit", ".mrc", ".map"]:
            lo_map = Dmap(self.input_map, isovalue=self.isovalue)
            lo_map.reduce_void()
            self.voxsp = lo_map.voxsp
            self.processed_map = os.path.join(init_path, "%s_mad.mrc"%self.map_name)
            lo_map.write_to_mrc(self.processed_map)

        elif os.path.splitext(self.input_map)[-1] in [".pdb"]:
            print("MaD> PDB provided for density map: %s"%(self.input_map))
            print("     Simulating at specified resolution and voxel spacing of 1.2 angstroms")
            self.voxsp = 1.2
            pdb_struct = PDB(self.input_map)
            self.processed_map = os.path.join(init_path, "%s_simulated_map.mrc"%self.map_name)
            pdb_struct.structure_to_density(self.resolution, self.voxsp, outname=self.processed_map)
        else:
            print("MaD> ERROR: density map not understood (either sit/mrc/map format or PDB for simulated density): %s"%self.input_map)

        # > subunits
        for k in self.input_subunits.keys():
            pdb_file, n_copies = self.input_subunits[k]
            out_name = os.path.join(init_path, "%s.pdb"%(k))
            move_copy_structure(pdb_file, out_name, transform=self.transform_subunits)
            self.processed_subunits[k] = [out_name, n_copies]

        # > ensembles
        for ek in self.input_ensembles.keys():
            ensemble = self.input_ensembles[ek]
            self.processed_ensembles[ek] = {}
            for fk in ensemble.keys():
                pdb_file, n_copies = ensemble[fk]
                nopath_pdb_file = os.path.split(pdb_file)[-1]
                processed_pdb_file = os.path.join(init_path, nopath_pdb_file)
                move_copy_structure(pdb_file, processed_pdb_file, transform=self.transform_subunits)
                self.processed_ensembles[ek][fk] = [processed_pdb_file, n_copies]


    def _describe_struct(self, struct, detect_sigma, presmooth_sigma, ori_eqsp_size, dsc_eqsp_size, dsc_subregions, patch_size):
        ms = MapSpace(struct, resolution=self.resolution, voxelsp=self.voxsp, sig_init=detect_sigma, sig_presmooth=presmooth_sigma)
        det = Detector()
        ori = Orientator(ori_radius = patch_size)
        dsc = Descriptor(dsc_radius = patch_size)

        ms.build_space()
        anchors = det.find_anchors(ms)
        oriented_anchors = ori.assign_orientations(ms, anchors)
        descriptors = dsc.generate_descriptors(ms, oriented_anchors)
        return descriptors


    def _match_filter_refine(self, pdbfile, n_copies, k, cc_threshold, weight_threshold, n_samples):
        # Adjust the amount of descriptor pairs according to the expected number of copies of that subunit
        n_samples_sub = n_samples * n_copies

        # Matching descriptors
        print("MaD> Matching descriptors (%s vs. %s) (cc = %.2f)..."%(self.map_name, k, cc_threshold))

        # If dealing with an ensemble, we reload descriptors to be nice to Miss Memory
        if isinstance(self.dsc_dict[k], str):
            match_results, map_anchors, comp_anchors = self._match_dsc(self.map_dsc, self._load_descriptors(self.dsc_dict[k]), cc_threshold=cc_threshold)
        else:
            match_results, map_anchors, comp_anchors = self._match_dsc(self.map_dsc, self.dsc_dict[k], cc_threshold=cc_threshold)


        # Keeping in case.
        # This section saves matching results to a file to avoid repeating the procedure.
        # match_outname = os.path.join(self.out_folder, "matching_%s_cc%.2f_w%i_samples%i.h5"%(k, cc_threshold, weight_threshold, n_samples_sub))
        # if not os.path.exists(match_outname):
        #     print("MaD> Matching descriptors (%s vs. %s) (cc = %.2f)..."%(self.map_name, k, cc_threshold))

        #     # If dealing with an ensemble, we reload descriptors to be nice to Miss Memory
        #     if isinstance(self.dsc_dict[k], str):
        #         match_results, map_anchors, comp_anchors = self._match_dsc(self.map_dsc, self._load_descriptors(self.dsc_dict[k]), cc_threshold=cc_threshold)
        #     else:
        #         match_results, map_anchors, comp_anchors = self._match_dsc(self.map_dsc, self.dsc_dict[k], cc_threshold=cc_threshold)
        #         # self._save_matching(match_outname, match_results, map_anchors, comp_anchors)
        # else:
        #     match_results, map_anchors, comp_anchors = self._load_matching(match_outname)
        #     print("MaD> Loaded matched descriptor pairs (%s vs. %s) (cc = %.2f)..."%((self.map_name, k), k, cc_threshold))

        print("MaD> Filtering descriptor pairs (map %s vs. structure %s) (weight=%i, n_samples=%i*%i)..."%(self.map_name, k, weight_threshold, n_samples, n_copies))
        filtered_candidate_list = self._filter_dsc_pairs(pdbfile, match_results, map_anchors, comp_anchors, wthresh=weight_threshold, n_samples=int(n_samples_sub))

        # If curious, save solutions at this state in the "pre_solutions" folder
        #self._save_solutions_from_filtered(filtered_candidate_list, k)

        print("MaD> Refining %s in %s..."%(self.map_name, k))
        refined_candidate_list = self._refine_filtered_solutions(pdbfile, filtered_candidate_list, map_anchors, comp_anchors)
        solutions_filename_list = self._save_solutions_refined(refined_candidate_list, k)

        return solutions_filename_list


    def _match_dsc(self, lo_dsc_list, hi_dsc_list, anchor_dist_thresh=4, cc_threshold=0.65):
        # Get descriptor vectors + normalize
        hi_vec_ar = np.array([df.lin_ar_subeqsp / np.linalg.norm(df.lin_ar_subeqsp) if np.linalg.norm(df.lin_ar_subeqsp) > 0 else df.lin_ar_subeqsp for df in hi_dsc_list])
        lo_vec_ar = np.array([df.lin_ar_subeqsp / np.linalg.norm(df.lin_ar_subeqsp) if np.linalg.norm(df.lin_ar_subeqsp) > 0 else df.lin_ar_subeqsp for df in lo_dsc_list])

        # Match with cross-correlation
        preds = np.dot(hi_vec_ar, lo_vec_ar.T)

        # Get pairs over threshold
        pair_generator = zip(*np.where(preds > cc_threshold))
        pair_list = [tuple(p) for p in pair_generator]

        # Get anchor coordinates from putative matching anchors
        hi_mapcoords = np.unique([hi_dsc_list[p[0]].subv_map_coords for p in pair_list], axis=0)
        lo_mapcoords = np.unique([lo_dsc_list[p[1]].subv_map_coords for p in pair_list], axis=0)

        # Process pairs
        lo_tree = cKDTree(lo_mapcoords)
        results = []
        for phi, plo in pair_list:
            mhi = hi_dsc_list[phi]
            mlo = lo_dsc_list[plo]

            # Get other anchors
            R = np.dot(np.linalg.inv(mlo.Rfinal), mhi.Rfinal)
            l = hi_mapcoords.shape[0]
            cur_hi_mapcoords = hi_mapcoords - mhi.subv_map_coords

            # Rotate, translate cloud
            cur_hi_mapcoords = np.dot(cur_hi_mapcoords, R.T)
            cur_hi_mapcoords = cur_hi_mapcoords + mlo.subv_map_coords

            # Get repeatability
            distances, indices = lo_tree.query(cur_hi_mapcoords, distance_upper_bound=anchor_dist_thresh)
            repeatability = 100 * np.count_nonzero(distances < anchor_dist_thresh) / l

            # Record results. CC between descr, repeatability, index/octave, coords and rotation matrix
            results.append(np.concatenate([[preds[phi, plo], repeatability, mlo.index, mlo.oct_scale, mlo.main_bin, mhi.index, mhi.oct_scale, mhi.main_bin], mhi.subv_map_coords, mlo.subv_map_coords, R.flatten()]))

        return results, lo_mapcoords, hi_mapcoords


    def _filter_dsc_pairs(self, pdbfile, match_data, lo_cloud, hi_cloud, wthresh=4, n_samples=200):
        # Matching data
        # > Format:
        #     0        1       2-4                5-7            8-10       11-13      14-20
        #   d_CC    repeat  lo: idx,oct,bin  hi: idx,oct,bin   hi_coord   hi_coord    Rmatrix
        names = [""]*100
        d_CC = 0;      names[d_CC]   =  "Desc-CC"
        repeat  = 1;   names[repeat] =  "Repeatability"
        lo_idx = 2;    names[lo_idx] =  "ref idx"
        lo_oct = 3;    names[lo_oct] =  "ref octave"
        lo_bin = 4;    names[lo_bin] =  "ref bin"
        hi_idx = 5;    names[hi_idx] =  "sub idx"
        hi_oct = 6;    names[hi_oct] =  "sub octave"
        hi_bin = 7;    names[hi_bin] =  "sub bin"
        hi_coord = slice(8,11)
        lo_coord = slice(11,14)
        R1 = slice(14,17)
        R2 = slice(17,20)
        R3 = slice(20,23)

        # RMSD between point clouds to cluster together
        rmsdcloud_thresh = 10

        # Load matching data + sort by repeatability
        data_sorted_repeat = np.array(sorted(match_data, key=itemgetter(repeat), reverse=True))

        # High-res PDB to move around
        chain = PDB(pdbfile)
        chain_init = chain.get_coords().copy()

        # Point cloud of best solution
        init_hi_cloud = hi_cloud.copy()
        best_s = data_sorted_repeat[0]
        R_hicloud = np.array([best_s[R1], best_s[R2], best_s[R3]])
        hi_cloud = init_hi_cloud - best_s[hi_coord]
        hi_cloud = np.dot(hi_cloud, R_hicloud.T)
        hi_cloud = hi_cloud + best_s[lo_coord]

        # Prime candidates
        candidate_list = [0]
        cand_cloud_list = [hi_cloud]
        cand_weight_dic = {0:1}
        candidate_anchor_dic = {0:[[best_s[hi_coord], best_s[lo_coord], best_s[hi_bin], best_s[lo_bin]]]}

        # Clustering hierarchical with repeatability ordering
        sol_counter = 1
        for s in data_sorted_repeat[1:n_samples]:
            # Move point cloud
            cur_hicloud = init_hi_cloud - s[hi_coord]
            cur_hicloud = np.dot(cur_hicloud, np.array([s[R1], s[R2], s[R3]]).T)
            cur_hicloud = cur_hicloud + s[lo_coord]

            # Get RMSD between candidate point clouds and current solution
            cloudsq = np.square(cand_cloud_list - cur_hicloud)
            rmsd_cloud_list = np.sqrt(np.sum(cloudsq, axis=(1,2)) / len(cur_hicloud))

            # Add weight to the closest candidate if low enough, otherwise create a new candidate
            if np.amin(rmsd_cloud_list) > rmsdcloud_thresh:
                candidate_list.append(sol_counter)
                cand_cloud_list.append(cur_hicloud)
                cand_weight_dic[sol_counter] = 1
                candidate_anchor_dic[sol_counter] = [[s[hi_coord], s[lo_coord], s[hi_bin], s[lo_bin]]]
            else:
                cand_weight_dic[candidate_list[np.argmin(rmsd_cloud_list)]] += 1
                candidate_anchor_dic[candidate_list[np.argmin(rmsd_cloud_list)]].append([s[hi_coord], s[lo_coord], s[hi_bin], s[lo_bin]])
            sol_counter += 1

        # Get solutions
        # > Repeatability threshold
        # > Weight threshold (clustered solutions #)
        rep_thresh = max(5, best_s[repeat] * 0.3)

        filtered_candidates = []
        sol_counter = 0
        for idx, cand in enumerate(candidate_list):
            s = data_sorted_repeat[cand]
            weight = cand_weight_dic[cand]

            if weight < wthresh or s[repeat] < rep_thresh:
                continue

            # Move structure and save pre-refinement solution
            chain.set_coords(chain_init)
            chain.translate_atoms(-s[hi_coord]) # Chain was rotated around keypoint, so put it on origin
            chain.rotate_atoms(np.array([s[R1], s[R2], s[R3]]).T)
            chain.translate_atoms(s[lo_coord]) # with subv localiz

            # Get lo anchors from cluster that has matching dsc
            clustered_anchors = candidate_anchor_dic[cand]

            # Solution validated; add
            filtered_candidates.append([s[hi_coord], s[lo_coord], np.array([s[R1], s[R2], s[R3]]).T, s[d_CC], weight, s[repeat], s[repeat]*weight, deepcopy(chain), clustered_anchors])
            sol_counter += 1

        # Sort by weight * repeatability
        filtered_candidates = sorted(filtered_candidates, key=itemgetter(6), reverse=True)

        return filtered_candidates


    def _refine_filtered_solutions(self, pdbfile, filtered_candidate_list, lo_cloud, hi_cloud):
        # High-res PDB to move around
        hi_pdb = PDB(pdbfile)
        hi_init = hi_pdb.get_coords().copy()

        # Load density map
        dmap = Dmap(self.processed_map)

        # Refine solutions, update repeatability
        refined_candidates_list = []
        kdtree = cKDTree(lo_cloud)
        for idx, cand in enumerate(filtered_candidate_list):
            hi_coord, lo_coord, R, cc, weight, repeat, score1, solution_fname, clustered_anchors = cand

            # Move structure and refine fitting
            hi_pdb.set_coords(hi_init)
            hi_pdb.translate_atoms(-hi_coord) # Chain was rotated around keypoint, so put it on origin
            hi_pdb.rotate_atoms(R)
            hi_pdb.translate_atoms(lo_coord) # with subv localiz

            # Refine fit
            rmsd_beforeAfter, converged, step = refine_pdb(dmap, hi_pdb, n_steps=500, max_step_size=1, min_step_size=0.1)

            # Update repeatability
            R, T = get_rototrans_SVD(hi_init, hi_pdb.coords)
            s_a = np.dot(hi_cloud, R) + T
            distances, indices = kdtree.query(s_a, distance_upper_bound=dmap.voxsp*1.5)
            repeatability = 100 * np.count_nonzero(distances < dmap.voxsp*2) / hi_cloud.shape[0]

            # Matching anchors
            corresp_anchors_hi = s_a[distances < dmap.voxsp*2]

            # Save results
            if repeatability > 0:
                refined_candidates_list.append([deepcopy(hi_pdb), corresp_anchors_hi, repeatability, weight, clustered_anchors])

        # Get final unique results
        final_solutions = []
        for cand_idx, cand in enumerate(refined_candidates_list):
            hi_pdb, corresp_anchors, repeatability, weight, clustered_anchors = cand
            if not len(final_solutions):
                # Get CCC for best solution
                sub_grid, sx, sy, sz = hi_pdb.structure_to_density(self.resolution, dmap.voxsp)
                ccc = dmap.get_CCC_with_grid(sub_grid, sx, sy, sz)

                final_solutions.append([hi_pdb, corresp_anchors, repeatability, weight, ccc, clustered_anchors])
            else:
                # Check RMSD with recorded final solutions
                rmsd_list = []
                for fsol in final_solutions:
                    rmsd_list.append(cand[0].get_rmsdCA_with(fsol[0]))

                # We have a clone; add weight to the better solution
                if np.min(rmsd_list) < 6:
                    final_solutions[np.argmin(rmsd_list)][3] += weight
                    [final_solutions[np.argmin(rmsd_list)][5].append(x) for x in clustered_anchors]
                    continue

                # We have a unique solution
                else:
                    # Get CCC
                    sub_grid, sx, sy, sz = hi_pdb.structure_to_density(self.resolution, dmap.voxsp)
                    ccc = dmap.get_CCC_with_grid(sub_grid, sx, sy, sz)

                    final_solutions.append([hi_pdb, corresp_anchors, repeatability, weight, ccc, clustered_anchors])

        # Sort results with new score
        for idx, sol in enumerate(final_solutions):
            hi_pdb, corresp_anchors, repeat, weight, ccc, clustered_anchors = sol
            super_score = repeat * weight * ccc
            sol.append(super_score)

        final_solutions = sorted(final_solutions, key=itemgetter(-1), reverse=True)
        return final_solutions


    def _build_from_single(self, sub_key, homomultimer=False):
        if homomultimer:
            comb_sol_path = os.path.join(self.out_folder, "assembly_models")
        else:
            comb_sol_path = os.path.join(self.out_folder, "subcomplexes")
        if not os.path.exists(comb_sol_path):
            os.mkdir(comb_sol_path)

        n_copies, solution_list = self.buildable_subunits[sub_key]

        # Properly set amount of solutions to combine
        if n_copies > len(solution_list):
            print("MaD> Not enough solutions to cover all copies for subunit %s !"%sub_key)
            print("     Maybe try increasing n_samples or reducing min_cc/wthresh ?")
            print("     In the meantime, trying with available solutions...")
            n_copies = len(solution_list)

        # Find candidate assemblies: elements contain idx of solutions, std / max overlap between structs
        candidate_list = []

        # > If single copy in complex, just save subcomplexes from original solutions.
        if n_copies == 1:
            candidate_list = [[tuple([s_idx]), 0, 0, 0] for s_idx in range(len(solution_list))]
        else:
            # Get grids for overlap and prep for repeated structures
            map_list = []
            for idx, sol in enumerate(solution_list):
                grid, xi, yi, zi = PDB(sol).structure_to_density(5, 2, isovalue=0.2) #resolution 1, voxelsp 2. Low res prevent biasing overlap
                map_list.append([idx, [grid, xi, yi, zi]])

            # Pre-compute overlaps between all pairs
            n_sol = len(solution_list)
            overlap_pairs = np.zeros((n_sol, n_sol))
            for c in combinations(map_list, 2):
                m1, m2 = c
                overlap = get_overlap(m1[1], m2[1], 2)
                overlap_pairs[m1[0], m2[0]] = overlap

            # Show overlap table
            print("MaD> Pairwise overlaps between solutions of %s:"%sub_key)
            print()
            nspaces_overlap_pairs = len(str(len(overlap_pairs)))
            for idx, s in enumerate(overlap_pairs):
                row = "%s%s | "%('.'.join([str(idx), sub_key]), " "*(nspaces_overlap_pairs - len(str(idx))))
                for v in s:
                    if v == 0.0:
                        row = row + "   0  "
                    else:
                        row = row + "%.3f "%v
                print(row)
            print()

            # For each candidate assembly, get total overlap
            print("MaD> Assembling %i copies of chain %s from %i solutions..."%(n_copies, sub_key, n_sol))
            for c in combinations(range(n_sol), n_copies):
                data_overlap = [overlap_pairs[ic[0], ic[1]] for ic in combinations(c, 2)]
                max_overlap = np.max(data_overlap)
                std_overlap = np.std(data_overlap)
                sum_overlap = np.sum(data_overlap) / n_copies
                candidate_list.append([c, sum_overlap, std_overlap, max_overlap])

            # Sort by increasing overlap sum
            candidate_list = sorted(candidate_list, key=itemgetter(3))

        # Save subcomplexes of current subunit
        if not homomultimer:
            valid_candidates_list = []
            for s_idx, s in enumerate(candidate_list):
                sub_idx, score_sum, score_std, score_max = s
                if score_max > self.max_overlap_complex:
                    continue

                ass_code = "_".join(["%s%i"%(sub_key, x) for x in sub_idx])
                comp_outname = os.path.join(comb_sol_path, "SubComplex%s_%i_%s.pdb"%(sub_key, s_idx, ass_code))
                self._write_complex_from_components([solution_list[idx] for idx in sub_idx], comp_outname)
                valid_candidates_list.append(comp_outname)

            if n_copies > 1:
                print("MaD> Generated %i subcomplexes from component %s"%(len(valid_candidates_list), sub_key))
            return valid_candidates_list

        # Make final models for homomultimeric assemblies
        else:
            # Compute CC / make PDBs
            dmap = Dmap(self.processed_map)
            print("MaD> Final models docked in map %s: "%self.map_name)
            print()

            # Sort and display solutions
            header = "    # |   CC   | Sum(O) | Std(O) | Max(O) | Composition"
            header_csv = ["#", "CC", "Sum(O)", "Std(O)", "Max(O)", "Composition"]
            data_csv = []
            print(header)
            print("-" * len(header))
            for cnt, s in enumerate(candidate_list):
                sub_idx, score_sum, score_std, score_max = s
                sub_idx_str = [str(s) for s in sub_idx]

                if cnt >= self.max_models or (score_max > 0.1 and cnt):
                    break
                cnt += 1

                comp_outname = os.path.join(comb_sol_path, "Model_%i.pdb"%(cnt))
                self._write_complex_from_components([solution_list[idx] for idx in sub_idx], comp_outname)
                comp_pdb = PDB(comp_outname)
                g, i1, i2, i3 = comp_pdb.structure_to_density(4, dmap.voxsp)
                ccc = dmap.get_CCC_with_grid(g, i1, i2, i3)
                # ccc = 0

                print("  %3i | %6.2f  %6.2f   %6.2f   %6.2f  | %s"%(cnt, ccc, score_sum, score_std, score_max, ".".join(sub_idx_str)))
                data_csv.append([cnt, ccc, score_sum, score_std, score_max, sub_idx_str])
            print("-" * len(header))
            if len(data_csv):
                pd.DataFrame(data_csv).to_csv(os.path.join(self.out_folder, "complex_ranking.csv"), index=False, header=header_csv)


    def _build_models(self, sub_sol_dict):
        print("MaD> Building assembly models from %i components..."%len(sub_sol_dict))

        map_dic = {}

        sol_counter = 0
        sol_counter_list = []
        comp_sol_list = []
        comp_sol_file_list = []
        map_list_for_chain_id = []
        sol_chain_id_list = []
        for sub_key, sol_list in sub_sol_dict.items():
            sol_counter_l = []
            for sol in sol_list:
                grid, xi, yi, zi = PDB(sol).structure_to_density(5, 2, isovalue=0.2)
                map_list_for_chain_id.append([grid, xi, yi, zi])
                comp_sol_list.append([sub_key, sol_counter])
                comp_sol_file_list.append(sol)
                sol_chain_id_list.append(sub_key)
                sol_counter_l.append(sol_counter)
                sol_counter += 1
            sol_counter_list.append(sol_counter_l)
            map_dic[sub_key] = map_list_for_chain_id

        # Pre-compute overlaps between all pairs
        n_sol = len(comp_sol_list)
        overlap_pairs = np.zeros((n_sol, n_sol))
        for c in combinations(comp_sol_list, 2):
            m1, m2 = c
            overlap = get_overlap(map_dic[m1[0]][m1[1]], map_dic[m2[0]][m2[1]], 2)
            overlap_pairs[m1[1], m2[1]] = overlap

        # Show overlap table
        print("MaD> Pairwise overlaps between solutions of %s:"%sub_key)
        print()
        nspaces_overlap_pairs = len(str(len(overlap_pairs)))
        max_str_len = max([len(c[0]) for c in (comp_sol_list)])
        for idx, s in enumerate(overlap_pairs):
            sol_idx = comp_sol_list[idx][0]
            row = "%5s%s%s | "%('.'.join([str(idx), sol_idx]), " "*(nspaces_overlap_pairs - len(str(idx))),  " "*(max_str_len - len(comp_sol_list[idx][0])))

            for v in s:
                if v == 0.0:
                    row = row + "   0  "
                else:
                    row = row + "%.3f "%v
            print(row)
        print()

        # Get overlap results between each component pair
        solution_assemblies = []
        for c in cartesian(sol_counter_list):
            X, Y = np.meshgrid(c, c)
            data_overlap = overlap_pairs[X.ravel(),Y.ravel()]
            max_overlap = np.max(data_overlap)
            std_overlap = np.std(data_overlap)
            sum_overlap = np.sum(data_overlap)
            solution_assemblies.append([c, sum_overlap, std_overlap, max_overlap])

        # Sort solutions
        solution_assemblies = sorted(solution_assemblies, key=itemgetter(1))
        ass_model_path = os.path.join(self.out_folder, "assembly_models")
        if not os.path.exists(ass_model_path):
            os.mkdir(ass_model_path)

        # Compute CC / make PDBs
        dmap = Dmap(self.processed_map)

        print("MaD> Final models docked in map %s: "%self.map_name)
        print()

        # Sort and display solutions
        header = "    # |   CC   | Sum(O) | Std(O) | Max(O) | Composition"
        header_csv = ["#", "CC", "Sum(O)", "Std(O)", "Max(O)", "Composition"]
        dashes = "-" * len(header)
        print(header)
        print(dashes)
        data_csv = []
        for cnt, s in enumerate(solution_assemblies):
            sub_idx, score_sum, score_std, score_max = s
            sub_idx_str = [str(s) for s in sub_idx]
            if cnt >= self.max_models or (score_max > self.max_overlap_complex and cnt):
                break
            cnt += 1

            comp_outname = os.path.join(self.out_folder, "assembly_models", "Model_%i.pdb"%(cnt))
            self._write_complex_from_components([comp_sol_file_list[idx] for idx in sub_idx], comp_outname)
            comp_pdb = PDB(comp_outname)
            g, i1, i2, i3 = comp_pdb.structure_to_density(4, dmap.voxsp)
            ccc = dmap.get_CCC_with_grid(g, i1, i2, i3)
            # ccc = 0
            print("  %3i | %6.2f  %6.2f   %6.2f   %6.2f  | %s"%(cnt, ccc, score_sum, score_std, score_max, ".".join(sub_idx_str)))
            data_csv.append([cnt, ccc, score_sum, score_std, score_max, sub_idx_str])
        print("-" * len(header))
        if len(data_csv):
            pd.DataFrame(data_csv).to_csv(os.path.join(self.out_folder, "complex_ranking.csv"), index=False, header=header_csv)


######## I/O ########

    def _save_descriptors(self, df_list, outname):
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

    def _load_descriptors(self, input_name):
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
        return df_list

    def _save_matching(self, outname, match_results, map_anchors, comp_anchors):
        hf = h5py.File(outname, 'w')
        hf.create_dataset('match', data=match_results)
        hf.create_dataset('map_anchors', data=map_anchors)
        hf.create_dataset('comp_anchors', data=comp_anchors)
        hf.close()

    def _load_matching(self, input_name):
        hf = h5py.File(input_name, 'r')
        match_results = hf.get('match')
        map_anchors = hf.get('map_anchors')
        comp_anchors = hf.get('comp_anchors')
        return match_results, map_anchors, comp_anchors

    def _save_solutions_from_filtered(self, filtered_candidate_list, sub_key):
        # Save solutions before refinement
        sol_path = os.path.join(self.out_folder, "pre_solutions")
        if not os.path.exists(sol_path):
            os.mkdir(sol_path)

        header = "|   # |   dCC  | Repeat |   W |    R*W   |"
        header_csv = ["ID", "dCC", "Repeatability", "Weight", "RW"]
        sep =    "------------------------------------------"
        print("\n"+sep+"\n"+header+"\n"+sep)

        data_csv = []
        sol_filename_list = []
        for idx, cand in enumerate(filtered_candidate_list):
            cand_anchor_hi, cand_anchor_lo, _, cc, weight, repeat, score, sol_pdb, clustered_anchors = cand

            solution_fname = os.path.join(sol_path, "presol_%s_%i.pdb"%(sub_key, idx))
            sol_pdb.write_pdb(solution_fname)
            sol_filename_list.append(solution_fname)

            self._save_oriented_anchors_as_pdb(clustered_anchors, sol_path, "%s_%i"%(sub_key, idx))

            sol_str =  "| %3i |  %5.3f |  %5.2f | %3i |  %7.2f |"%(idx,  cc, repeat, weight, score)
            data_csv.append([idx, cc, repeat, weight, score])
            print(sol_str)
        print(sep+"\n")
        if len(data_csv):
            pd.DataFrame(data_csv).to_csv(os.path.join(self.out_folder, "Solutions_filtered_%s.csv"%sub_key), index=False, header=header_csv)

        return sol_filename_list


    def _save_solutions_refined(self, refined_solutions, sub_key):
        # Save solutions after refinement
        sol_path = os.path.join(self.out_folder, "individual_solutions")
        if not os.path.exists(sol_path):
            os.mkdir(sol_path)

        anchor_path = os.path.join(sol_path, "anchor_files")
        if not os.path.exists(anchor_path):
            os.mkdir(anchor_path)

        header = "|  # | Repeat | Weight |   mCC  |  RWmCC |"
        header_csv = ["ID", "Repeatability", "Weight", "mCC", "RWmCC"]
        sep =    "------------------------------------------"

        print("\n"+sep+"\n"+header+"\n"+sep)
        data_csv = []
        sol_filename_list = []
        for idx, sol in enumerate(refined_solutions):
            hi_pdb, corresp_anchors, repeat, weight, ccc, clustered_anchors, super_score = sol

            solution_fname = os.path.join(sol_path, "sol_%s_%i.pdb"%(sub_key, idx))
            hi_pdb.write_pdb(solution_fname)
            sol_filename_list.append(solution_fname)

            corresp_fname = os.path.join(anchor_path, "corresp_anchors_%s_%i.pdb"%(sub_key, idx))
            self._save_coords_as_pdb(corresp_anchors, corresp_fname)
            self._save_oriented_anchors_as_pdb(clustered_anchors, anchor_path, "%s_%i"%(sub_key, idx))

            sol_str =  "| %2i | %6.2f | %6i | %6.2f | %6.2f |"%(idx, repeat, weight, ccc, super_score)
            data_csv.append([idx, repeat, weight, ccc, super_score])
            print(sol_str)
        print(sep+"\n")
        if len(data_csv):
            pd.DataFrame(data_csv).to_csv(os.path.join(self.out_folder, "Solutions_refined_%s.csv"%sub_key), index=False, header=header_csv)

        return sol_filename_list


    def _write_complex_from_components(self, components, outname):
        occupancy = 1.0
        tempFactor = 0.0
        fout = open(outname, "w")
        chain = "@" # @ is before "A" in ASCII; we reset chain order for simplicity
        for comp in components:
            pdb = PDB(comp)
            for i in range(pdb.n_atoms):
                if pdb.info[i][0] == 1:
                    chain = chr(ord(chain)+1)
                    if chain != "A":
                        fout.write("TER\n")

                # Check for atom type: if 4 letters long, start in column 13 otherwise no space.
                # Else, start in column 14. Differs from format convention v3.30 but always see it this way in files.
                if len(pdb.info[i][1]) == 4:
                    line_model = "%-6s%5i %-4s %3s%2s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s"
                else:
                    line_model = "%-6s%5i  %-3s %3s%2s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s"
                line = line_model%(pdb.info[i][6], pdb.info[i][0], pdb.info[i][1], pdb.info[i][2], chain, pdb.info[i][4], pdb.coords[i][0], pdb.coords[i][1], pdb.coords[i][2], occupancy, tempFactor, pdb.info[i][5])
                fout.write(line+"\n")
        fout.close()


    def _save_coords_as_pdb(self, coords, outname):
        # Create dummy atoms for keypoint vor visualization in VMD or similar
        pdb_data = []
        for idx in range(len(coords)):
            atom_type = 'O'
            atom_res = "EPC"
            atom_chain = 'E'

            tmp_atom = []
            tmp_atom.append(idx) #=atom_nb
            tmp_atom.append(atom_type)
            tmp_atom.append(atom_res)
            tmp_atom.append(atom_chain)
            tmp_atom.append(idx)
            tmp_atom.append(coords[idx][0])
            tmp_atom.append(coords[idx][1])
            tmp_atom.append(coords[idx][2])

            tmp_atom.append(idx/len(coords))
            pdb_data.append(tmp_atom)

        with open(outname,"w") as f:
            for x in pdb_data:
                # Check for atom type: if 4 letters long, start in column 13 otherwise no space.
                # Else, start in column 14. Differs from format convention v3.30 but always see it this way in files.

                line_model = "%-6s%5i  %-3s %3s%2s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s"
                line = line_model%("ATOM", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1, x[8], "O")

                f.write(line+"\n")

    def _save_oriented_anchors_as_pdb(self, anchor_list, anchor_path, identifier):
        # anchor_list contains anchors from hi and lo that have matching dsc. Coordinates + bin of main dominant orientation.
        # > save PDB with coords as oxygen for anchor coords, and nitrogen for the direction of the domi orientation

        # make eqsp to get cartesian coords of the center of the dominant zone
        eqsp = EQSP_Sphere()

        for idx in [[0,2, "hi"], [1,3, "lo"]]:
            anchor_idx, bin_idx, target = idx

            # Create dummy atoms for visualization, and bld file for chimeraX to see directions
            pdb_data = []
            bld_data_ori = []
            bld_data_cor = []
            at_num = 1
            for anch_i, anch_ori in enumerate(anchor_list):
                coords_anc = anch_ori[anchor_idx]
                mbin = int(anch_ori[bin_idx])
                coords_ori = np.subtract(coords_anc, eqsp.c_center(mbin) * 10)

                # Coordinates of anchor
                tmp_atom = []
                tmp_atom.append(at_num); at_num += 1 #=atom_nb
                tmp_atom.append("C")
                tmp_atom.append("ANC") # anchor
                tmp_atom.append("A")
                tmp_atom.append(anch_i + 1)
                tmp_atom.append(coords_anc[0])
                tmp_atom.append(coords_anc[1])
                tmp_atom.append(coords_anc[2])
                tmp_atom.append(anch_i/len(anchor_list))
                tmp_atom.append("C")
                pdb_data.append(tmp_atom)

                # Coordinates of ori
                tmp_atom = []
                tmp_atom.append(coords_anc[0])
                tmp_atom.append(coords_anc[1])
                tmp_atom.append(coords_anc[2])
                tmp_atom.append(coords_ori[0])
                tmp_atom.append(coords_ori[1])
                tmp_atom.append(coords_ori[2])
                tmp_atom.append(anch_i/len(anchor_list))
                tmp_atom.append("O")
                bld_data_ori.append(tmp_atom)

                if anchor_idx == 0:
                    coords_cor = anch_ori[anchor_idx + 1]
                    tmp_atom = []
                    tmp_atom.append(coords_anc[0])
                    tmp_atom.append(coords_anc[1])
                    tmp_atom.append(coords_anc[2])
                    tmp_atom.append(coords_cor[0])
                    tmp_atom.append(coords_cor[1])
                    tmp_atom.append(coords_cor[2])

                    bld_data_cor.append(tmp_atom)

            with open(os.path.join(anchor_path, "anchor_%s_%s.pdb"%(target, identifier)), "w") as f:
                for x in pdb_data:
                    line_model = "%-6s%5i  %-3s %3s%2s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s"
                    line = line_model%("ATOM", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1, x[8], x[9])
                    f.write(line+"\n")

            with open(os.path.join(anchor_path, "anchor_ori_%s_%s.bld"%(target, identifier)), "w") as f:
                f.write(".color black\n")
                for x in bld_data_ori:
                    f.write(".arrow %f %f %f %f %f %f 0.2 1.0 0.75\n"%(x[0], x[1], x[2], x[3], x[4], x[5]))

            if anchor_idx == 0:
                with open(os.path.join(anchor_path, "anchor_cor_%s.bld"%(identifier)), "w") as f:
                    f.write(".color black\n")
                    for x in bld_data_cor:
                        f.write(".cylinder %f %f %f %f %f %f 0.1 \n"%(x[0], x[1], x[2], x[3], x[4], x[5]))
