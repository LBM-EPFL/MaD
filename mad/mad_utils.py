import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as mplot
matplotlib.rcParams.update({'font.size': 16})

from copy import deepcopy
from shutil import copyfile
from operator import itemgetter
from scipy.spatial import cKDTree
from itertools import combinations
from sklearn.utils.extmath import cartesian


from ..descriptor.MapSpace import MapSpace
from ..descriptor.Detector import Detector
from ..descriptor.Orientator import Orientator
from ..descriptor.Descriptor import Descriptor
from ..descriptor.DensityFeature import DensityFeature
from ..structure.PDB import PDB
from ..structure.Dmap import Dmap
from ..eqsp.eqsp import EQSP_Sphere
from .structure_utils import move_copy_structure, refine_pdb
from .math_utils import get_rototrans_SVD

def setup(exp_map, resolution, isovalue, components_dict, patch_size=16, prefix_name=""):
    '''
    Checks file consistency, create results folder for docking results, and sub folder with processed input structures

    Parameters
    ----------
    exp_map : str
        path to density map file (MRC, MAP, SIT, SITUS formats)
    resolution : float
        resolution of exp map
    isovalue : float
        isovalue to apply to the exp map
    components_dict : dict
        key is chain identifier, value is a list with: string or list of string for file names, and number of copy in assembly

    Returns
    -------
    str
        returns path to results folder and initial files

    '''
    # Check map
    assert os.path.exists(exp_map), "MaD> Exp. map not found: %s"%exp_map
    print("MaD> Found assembly map: %s"%(exp_map))
    low_name = os.path.split(exp_map)[-1].split('.')[0]

    # Check components
    n_comp = []
    for k in components_dict.keys():
        path = components_dict[k][0]
        multiplicity = components_dict[k][1]
        comp_name = os.path.splitext(os.path.split(path)[-1])[0]
        assert os.path.exists(path), "Path for components not valid: %s"%path
        if os.path.isdir(path):
            comp_list = [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".pdb" or x.endswith(".PDB"))]
            assert len(comp_list), "MaD> Path for component %s does not contain any valid structure: %s"%(k, path)
            print("MaD> Found %i PDB structures for component: %s"%(len(comp_list), k))
            for c in comp_list:
                print("     %s"%c)
            components_dict[k].append(comp_list)
            n_comp.append("%sx%i"%(comp_name, multiplicity) + "e")
        elif os.path.isfile(path):
            assert path.endswith(".pdb") or path.endswith(".PDB"), "Mad> ERROR: not a pdb file: %s"%path
            print("MaD> Found PDB structure for component: %s"%(path))
            components_dict[k].append("single")
            n_comp.append("%sx%i"%(comp_name, multiplicity) + "s")
        else:
            print("MaD> ERROR: neither folder or file (%s)"%path)
            return "", ""

    # Results folder
    comp_str = ".".join(n_comp)
    if prefix_name == "":
        out_folder = "results/%s_%s_%.3f_%.3f_patch%i"%(low_name, comp_str, resolution, isovalue, patch_size)
    else:
        out_folder = "results/%s_%s_%s_%.3f_%.3f_patch%i"%(prefix_name, low_name, comp_str, resolution, isovalue, patch_size)

    # Subfolder with pre-processed structures
    init_path = os.path.join(out_folder, "initial_files")
    if not os.path.exists(out_folder):
        print("MaD> Created new folder for output files: %s"%out_folder)
        os.mkdir(out_folder)
        os.mkdir(init_path)
    else:
        # print("MaD> Results folder exist; if you want to repeat docking, delete it and try again")
        return out_folder, init_path

    # Exp map
    # Load map, apply isovalue, remove void, and save to folder
    if os.path.splitext(exp_map)[-1] in [".sit", ".mrc", ".map"]:
        low_map = Dmap(exp_map, isovalue=isovalue)
        low_map.reduce_void()

        if prefix_name == "":
            out_name = os.path.join(init_path, "%s_exp.mrc"%low_name)
        else:
            out_name = os.path.join(init_path, "%s_%s_exp.mrc"%(prefix_name, low_name))

        low_map.write_to_mrc(out_name)
        voxsp = low_map.voxsp
    elif os.path.splitext(exp_map)[-1] in [".pdb"]:
        print("MaD> PDB provided for density map: %s"%(exp_map))
        print("     Simulating at specified resolution and voxel spacing of 1.2 angstroms")
        voxsp = 1.2
        refpdb = PDB(exp_map)
        if prefix_name == "":
            out_name = os.path.join(init_path, "%s_simulated_map.mrc"%low_name)
        else:
            out_name = os.path.join(init_path, "%s_%s_simulated_map.mrc"%(prefix_name, low_name))
        refpdb.structure_to_density(resolution, voxsp, outname=out_name)
    else:
        print("MaD> ERROR: density map not understood (either sit/mrc/map format or PDB for simulated density): %s"%exp_map)

    # Copy components to folder and rename them (chain ID, multiplicity)
    # Also moves them to the origin and rotates them randomly
    multiplicity = []
    for k in components_dict.keys():
        struct_id = os.path.splitext(os.path.split(components_dict[k][0])[-1])[0]
        if components_dict[k][-1] == "single":
            if prefix_name == "":
                out_name = os.path.join(init_path, "%s_%s_%i.pdb"%(struct_id, k, components_dict[k][1]))
            else:
                out_name = os.path.join(init_path, "%s_%s_%s_%i.pdb"%(prefix_name, struct_id, k, components_dict[k][1]))
            move_copy_structure(components_dict[k][0], out_name)
            multiplicity.append([k, components_dict[k][1], "single"])
        else:
            for idx, c in enumerate(components_dict[k][-1]):
                move_copy_structure(c, os.path.join(init_path, "%s_%s%i_%i.pdb"%(struct_id, k, idx, components_dict[k][1])))
            multiplicity.append([k, components_dict[k][1], "ensemble"])

    # Save relevant parameters
    np.save(os.path.join(init_path, "parameters.npy"), np.array([resolution, voxsp, isovalue, patch_size, multiplicity], dtype=object))

    # Return paths to results folder and to initial pre-processed structures
    return out_folder, init_path

def get_descriptors(structure_folder):
    resolution, voxelsp, isovalue, patch_size, _ = np.load(os.path.join(structure_folder, "parameters.npy"), allow_pickle=True)
    dsc_dict = {"map": [], "comps": []}
    for struct in os.listdir(structure_folder):
        if os.path.splitext(struct)[-1] in [".map", ".mrc", ".pdb"]:
            # get name of structure and output name for dsc file
            if struct.endswith(".pdb"):
                struct_id = '_'.join(os.path.split(struct)[-1].split('_')[:-1])
                dsc_outname = "dsc_db/%s_patch%i_res%f_dsc.h5"%(struct_id, patch_size, resolution)
            else:
                struct_id = os.path.splitext(os.path.split(struct)[-1])[0]
                dsc_outname = "dsc_db/%s_patch%i_res%f_iso%f_dsc.h5"%(struct_id, patch_size, resolution, isovalue)

            # Generate descriptors if not present in database already
            if not os.path.exists(dsc_outname):
                descriptors = _describe_struct(os.path.join(structure_folder, struct), resolution, voxelsp, int(patch_size))
                save_descriptors(descriptors, dsc_outname)
            else:
                descriptors = load_descriptors(dsc_outname)
                print("MaD> %i descriptors for %s found in database"%(len(descriptors), struct_id))

            if struct.endswith(".pdb"):
                dsc_dict["comps"].append([os.path.join(structure_folder, struct), descriptors])
            else:
                dsc_dict["map"] = [os.path.join(structure_folder, struct), descriptors]
            print()
    return dsc_dict


def match_descriptors(dsc_dict, out_folder, min_cc=0.65, wthresh=4, n_samples=200, ref_pdb_filelist=[]):
    low_map, map_dsc = dsc_dict["map"]
    resolution, _, _, _, _ = np.load(os.path.join(out_folder, "initial_files", "parameters.npy"), allow_pickle=True)
    for i, [struct, comp_dsc] in enumerate(dsc_dict["comps"]):
        # Matching
        struct_id = struct_id = '_'.join(os.path.split(struct)[-1].split('_')[:-1])
        print("MaD> Matching descriptors (map vs. %s) (cc = %.2f)..."%(struct_id, min_cc))
        match_results, map_anchors, comp_anchors = _match_comp_and_map(map_dsc, comp_dsc, min_cc=min_cc)
        outname = os.path.join(out_folder, "matching_map_%s.h5"%struct_id)
        save_matching(outname, match_results, map_anchors, comp_anchors)

        # Filtering
        print("MaD> Filtering descriptor pairs (map vs. %s)..."%struct_id)
        filtered_candidate_list = _filter_dsc_pairs(out_folder, match_results, map_anchors, comp_anchors, struct, wthresh=wthresh, n_samples=int(n_samples), ref_pdb_filelist=ref_pdb_filelist)

        # Refinement
        print("MaD> Refining %s in map..."%struct_id)
        _refine_score(out_folder, filtered_candidate_list, map_anchors, comp_anchors, struct, low_map, resolution, ref_pdb_filelist=ref_pdb_filelist)


def build_assembly(out_folder):
    params = np.load(os.path.join(out_folder, "initial_files", "parameters.npy"), allow_pickle=True)
    res, voxsp, iso, patch_size, multi_ar = params
    mult_dict = {}
    for m in multi_ar:
        mult_dict[m[0]] = m[1]
    if len(multi_ar) == 1 and multi_ar[0] == 1:
        print("MaD> No assembly to build from a monomeric structure")
        return
    flist = os.listdir(os.path.join(out_folder, "individual_solutions"))

    # Get unique chain IDs (useful when ensembles are docked)
    state_id_list = []
    for f in flist:
        if f.startswith("sol_"):
            state_id_list.append(f.split('_')[-3])
    state_id_list = np.unique(state_id_list)

    chain_id_list = np.unique([x[0] for x in state_id_list])
    chain_dict = {}
    for chain in chain_id_list:
        chain_dict[chain] = []
    for s in state_id_list:
        chain_dict[s[0]].append(s)


    # Build complex from the single component
    if len(multi_ar) == 1:
        chain_id = chain_id_list[0]

        multiplicity = mult_dict[multi_ar[0][0]]
        _build_from_single(out_folder, chain_id, multiplicity, chain_dict, flist, final=True)

    # Or build subcomplexes if a component is present in > 1 copies
    else:
        sub_sol_list = []
        for chain_id in chain_id_list:
            multiplicity = mult_dict[chain_id]
            subcomplexes = _build_from_single(out_folder, chain_id, multiplicity, chain_dict, flist)
            sub_sol_list.append(subcomplexes)

        # Build final assembly models
        _build_models(out_folder, chain_id_list, sub_sol_list)

def score_ensembles(out_folder):
    # Find which components was given as ensemble
    params = np.load(os.path.join(out_folder, "initial_files/parameters.npy"), allow_pickle=True)
    for p in params[-1]:
        if p[-1] == "ensemble":
            struct_list = [[f.split("_")[1], os.path.join(out_folder, "Solutions_refined_%s.csv"%f.replace(".pdb", ""))] for f in os.listdir(os.path.join(out_folder, "initial_files")) if (f.endswith(".pdb") and f.split("_")[1][0] == p[0])]
            df_list = []
            for s in struct_list:
                df = pd.read_csv(s[1])
                df["StructID"] = [s[0]] * df.shape[0]
                df_list.append(df)
            all_sols = pd.concat(df_list)
            all_sols.sort_values(by="mCC", ascending=False)
            struct_id = all_sols["StructID"].unique()

            ranking = []
            rep_list = []
            wgt_list = []
            mCC_list = []
            rwc_list = []
            for s in struct_id:
                mean_rep = all_sols[all_sols["StructID"] == s]["Repeatability"].mean()
                mean_wgt = all_sols[all_sols["StructID"] == s]["Weight"].mean()
                mean_mCC = all_sols[all_sols["StructID"] == s]["mCC"].mean()
                mean_rwc = all_sols[all_sols["StructID"] == s]["RWmCC"].mean()
                ranking.append([s, mean_rep, mean_wgt, mean_mCC, mean_rwc])
                rep_list.append(mean_rep)
                wgt_list.append(mean_wgt)
                mCC_list.append(mean_mCC)
                rwc_list.append(mean_rwc)
            rep_ranking = sorted(ranking, key=itemgetter(1), reverse=True)
            wgt_ranking = sorted(ranking, key=itemgetter(2), reverse=True)
            mcc_ranking = sorted(ranking, key=itemgetter(3), reverse=True)
            rwc_ranking = sorted(ranking, key=itemgetter(4), reverse=True)
            print("MaD> Ranking for ensemble %s: "%p[0])
            print()
            print(" Score | Strucure ID (decreasing score)")
            print("----------------------------------------")
            rep = "  ".join(["%s-%.2f"%(cc[0], cc[1]) for cc in rep_ranking])
            wgt = "  ".join(["%s-%.2f"%(cc[0], cc[2]) for cc in wgt_ranking])
            mcc = "  ".join(["%s-%.2f"%(cc[0], cc[3]) for cc in mcc_ranking])
            rwc = "  ".join(["%s-%.2f"%(cc[0], cc[4]) for cc in rwc_ranking])
            print("      R  %s"%rep)
            print("      W  %s"%wgt)
            print("      C  %s"%mcc)
            print("    RWC  %s"%rwc)

        fig, axes = mplot.subplots(nrows=1, ncols=4, figsize=(12,5))
        colors = ["#a8e5cd", "#3dc6ea", "#229ced", "#1f92f1", "#3155c3", "#223f94", "#2f3c80"]
        for ax, l, n, c in zip(axes, [rep_list, wgt_list, mCC_list, rwc_list], ["Avg. R", "Avg. |clust|", "Avg. CC", "Avg. S"], colors):
            ax.bar(range(1,8), l, color=colors)
            ax.set_xticks(range(1,8))
            ax.set_xticklabels(["C%i"%i for i in range(1, 8)], rotation=90)
            ax.set_title(n)

        mplot.tight_layout()
        mplot.savefig("Plot_score_ensemble.png", dpi=600)

        fig, ax = mplot.subplots(nrows=1, ncols=1, figsize=(4,2))
        colors = ["#a8e5cd", "#3dc6ea", "#229ced", "#1f92f1", "#3155c3", "#223f94", "#2f3c80"]
        groel_rmsd = [6.571727548195097, 4.7999114064784, 4.686410092905678, 3.519410351453527, 1.3554497801884962, 3.6669198486397008, 4.515803741010494]

        ax.bar(range(1,8), groel_rmsd, color=colors)
        # ax.set_xticks([])
        ax.set_xticks(range(1,8))
        ax.set_xticklabels(["C%i"%i for i in range(1, 8)], rotation=90)
        ax.set_yticks([0, 2, 4, 6])
        ax.set_yticklabels(["0 Å ", "2 Å ", "4 Å ", "6 Å "])
        # ax.invert_yaxis()
        ax.set_title("RMSD with ref.")
        mplot.tight_layout()
        mplot.savefig("Plot_score_ensemble_rmsd.png", dpi=600)

        fig, ax = mplot.subplots(nrows=1, ncols=1, figsize=(4,2))
        ax.bar(range(1,8), rwc_list, color=colors)
        ax.set_xticks(range(1,8))
        ax.set_xticklabels(["C%i"%i for i in range(1, 8)], rotation=90)
        ax.set_title("Avg. MaD score")
        ax.set_yticks([0, 100, 200, 300])

        mplot.tight_layout()
        mplot.savefig("Plot_score_ensemble_rwc.png", dpi=600)




############################################
################# INTERNAL #################
############################################

def _describe_struct(struct, resolution, voxelsp, patch_size):
    ms = MapSpace(struct, resolution=resolution, voxelsp=voxelsp, sig_init=2)
    det = Detector()
    ori = Orientator(ori_radius = patch_size)
    dsc = Descriptor(dsc_radius = patch_size)

    ms.build_space()
    anchors = det.find_anchors(ms)
    oriented_anchors = ori.assign_orientations(ms, anchors)
    descriptors = dsc.generate_descriptors(ms, oriented_anchors)
    return descriptors

def _match_comp_and_map(lo_dsc_list, hi_dsc_list, anchor_dist_thresh=4, min_cc=0.65):
    # Get descriptor vectors + normalize
    hi_vec_ar = np.array([df.lin_ar_subeqsp / np.linalg.norm(df.lin_ar_subeqsp) if np.linalg.norm(df.lin_ar_subeqsp) > 0 else df.lin_ar_subeqsp for df in hi_dsc_list])
    lo_vec_ar = np.array([df.lin_ar_subeqsp / np.linalg.norm(df.lin_ar_subeqsp) if np.linalg.norm(df.lin_ar_subeqsp) > 0 else df.lin_ar_subeqsp for df in lo_dsc_list])

    # Match with cross-correlation
    preds = np.dot(hi_vec_ar, lo_vec_ar.T)

    # Get pairs over threshold
    pair_generator = zip(*np.where(preds > min_cc))
    pair_list = [tuple(p) for p in pair_generator]

    # Get anchor coordinates from putative matching anchors
    hi_mapcoords = np.unique([hi_dsc_list[p[0]].subv_map_coords for p in pair_list], axis=0)
    lo_mapcoords = np.unique([lo_dsc_list[p[1]].subv_map_coords for p in pair_list], axis=0)

    # Make KD tree on mapcoords; we only want a radius of anchors around chosen anchor
    # > The farther you get from anchor point, the more the structure shifts
    # > As a result, points are not repeatable beyond a given radius (or are repeatable when they should not)
    # > Uses radius of gyration of anchor point cloud to reduce anchor points considered at for each step
    # > => Complexity in O(N) except for building descr kdtree for each loop regardless of structure size
    # !! ISSUE WITH INDEXING SHIFTS DUE TO GETTING ANCHORS FROM MATCHING PAIRS ONLY
    # center = np.mean(hi_mapcoords, axis=0)
    # rgyr = np.sqrt(np.sum(np.sum((hi_mapcoords - center)**2, axis=1) / hi_mapcoords.shape[0]))
    # if rgyr > 40:
    #     masking_subcoords = True
    #     hi_tree = cKDTree(hi_mapcoords)
    #     mask_list = hi_tree.query_ball_point(hi_mapcoords, r=40)
    # else:
    #     masking_subcoords = False

    # Process pairs
    lo_tree = cKDTree(lo_mapcoords)
    results = []
    for phi, plo in pair_list:
        mhi = hi_dsc_list[phi]
        mlo = lo_dsc_list[plo]

        # Get other anchors, within rgyr of hi's anchor if masking is enabled
        R = np.dot(np.linalg.inv(mlo.Rfinal), mhi.Rfinal)
        # if masking_subcoords:
        #     l = hi_mapcoords[mask_list[mhi.index]].shape[0]
        #     cur_hi_mapcoords = hi_mapcoords[mask_list[mhi.index]] - mhi.subv_map_coords
        # else:
        l = hi_mapcoords.shape[0]
        cur_hi_mapcoords = hi_mapcoords - mhi.subv_map_coords

        # Rotate, translate cloud
        cur_hi_mapcoords = np.dot(cur_hi_mapcoords, R.T)
        cur_hi_mapcoords = cur_hi_mapcoords + mlo.subv_map_coords

        # Get repeatability
        # > With k=1, seems that query is faster than count_neighbours, query_ball_tree and similar other functions.
        distances, indices = lo_tree.query(cur_hi_mapcoords, distance_upper_bound=anchor_dist_thresh)
        repeatability = 100 * np.count_nonzero(distances < anchor_dist_thresh) / l

        # Record results. CC between descr, repeatability, index/octave, coords and rotation matrix
        results.append(np.concatenate([[preds[phi, plo], repeatability, mlo.index, mlo.oct_scale, mlo.main_bin, mhi.index, mhi.oct_scale, mhi.main_bin], mhi.subv_map_coords, mlo.subv_map_coords, R.flatten()]))

    return results, lo_mapcoords, hi_mapcoords

def _filter_dsc_pairs(out_folder, match_data, lo_cloud, hi_cloud, pdb_file, wthresh=4, n_samples=200, ref_pdb_filelist=[]):
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
    chain = PDB(pdb_file)
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

    # For comparison with existing structures
    ref_pdb_list = []
    if len(ref_pdb_filelist):
        for ref_pdbfile in ref_pdb_filelist:
            ref = PDB(ref_pdbfile)
            ref_pdb_list.append(ref)

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

        # If provided, check rmsd with structure list
        min_rmsd = np.inf
        min_idx = np.inf
        for idx, ref_pdb in enumerate(ref_pdb_list):
            rmsdca = chain.get_rmsdCA_with(ref_pdb)
            if min_rmsd > rmsdca:
                min_rmsd = rmsdca
                min_idx = idx
        rmsd = min_rmsd

        # Get lo anchors from cluster that has matching dsc
        clustered_anchors = candidate_anchor_dic[cand]

        # Solution validated; add
        filtered_candidates.append([s[hi_coord], s[lo_coord], np.array([s[R1], s[R2], s[R3]]).T, s[d_CC], weight, s[repeat], s[repeat]*weight, rmsd, min_idx, deepcopy(chain), clustered_anchors])
        sol_counter += 1

    # Sort by weight * repeatability
    filtered_candidates = sorted(filtered_candidates, key=itemgetter(6), reverse=True)

    # Save solutions before refinement
    if not os.path.exists(os.path.join(out_folder, "pre_solutions")):
        os.mkdir(os.path.join(out_folder, "pre_solutions"))
    sol_path = os.path.join(out_folder, "pre_solutions")
    hi_id = os.path.split(pdb_file)[-1].split('.')[0]
    out_filter = os.path.join(out_folder, "Solutions_filtered_%s.txt"%hi_id)
    sol_counter = 0
    with open(out_filter, "w") as fout:
        if len(ref_pdb_list):
            header = "|   # |   dCC  | Repeat |   W |    R*W   |  RMSD  |  Ref"
            sep =    "--------------------------------------------------------"
        else:
            header = "|   # |   dCC  | Repeat |   W |    R*W   |"
            sep =    "------------------------------------------"
        print("\n"+sep+"\n"+header+"\n"+sep)
        fout.write(sep+"\n")
        fout.write(header+"\n")
        fout.write(sep+"\n")
        for idx, cand in enumerate(filtered_candidates):
            cand_anchor_hi, cand_anchor_lo, _, cc, weight, repeat, score, rmsd, min_idx, sol_pdb, clustered_anchors = cand
            if len(ref_pdb_list):
                sol_str =  "| %3i |  %5.3f |  %5.2f | %3i |  %7.2f | %6.2f | %i "%(sol_counter,  cc, repeat, weight, score, rmsd, min_idx)
            else:
                sol_str =  "| %3i |  %5.3f |  %5.2f | %3i |  %7.2f |"%(sol_counter,  cc, repeat, weight, score)
            fout.write(sol_str+"\n")

            print(sol_str)

            solution_fname = os.path.join(sol_path, "presol_%s_%i.pdb"%(hi_id, sol_counter))
            sol_pdb.write_pdb(solution_fname)

            save_oriented_anchors_as_pdb(clustered_anchors, sol_path, "%s_%i"%(hi_id, sol_counter))

            sol_counter += 1

        print(sep+"\n")
        fout.write(sep+"\n")

    return filtered_candidates


def _refine_score(out_folder, filtered_candidate_list, lo_cloud, hi_cloud, pdb_file, low_map, res, ref_pdb_filelist=[]):
    # Make folder if not existing
    if not os.path.exists(os.path.join(out_folder, "individual_solutions")):
        os.mkdir(os.path.join(out_folder, "individual_solutions"))
    sol_path = os.path.join(out_folder, "individual_solutions")

    # High-res PDB to move around
    hi_id = os.path.split(pdb_file)[-1].split('.')[0]
    hi_pdb = PDB(pdb_file)
    hi_init = hi_pdb.get_coords().copy()

    # Reference PDBs if any
    ref_pdb_list = []
    if len(ref_pdb_filelist):
        for ref_pdbfile in ref_pdb_filelist:
            ref = PDB(ref_pdbfile)
            ref_pdb_list.append(ref)

    # Load density map
    dmap = Dmap(low_map)

    # Refine solutions, update repeatability
    refined_candidates_list = []
    kdtree = cKDTree(lo_cloud)
    for idx, cand in enumerate(filtered_candidate_list):
        hi_coord, lo_coord, R, cc, weight, repeat, score1, rmsd, _, solution_fname, clustered_anchors = cand

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
            # Get RMSD with original
            rmsd = np.inf
            min_idx = np.inf
            for idx, ref_pdb in enumerate(ref_pdb_list):
                rmsdca = hi_pdb.get_rmsdCA_with(ref_pdb)
                if rmsd > rmsdca:
                    rmsd = rmsdca
                    min_idx = idx

            # Get CCC for best solution
            sub_grid, sx, sy, sz = hi_pdb.structure_to_density(res, dmap.voxsp)
            ccc = dmap.get_CCC_with_grid(sub_grid, sx, sy, sz)

            final_solutions.append([hi_pdb, corresp_anchors, repeatability, weight, ccc, rmsd, min_idx, clustered_anchors])
        else:
            # Check RMSD with recorded final solutions
            rmsd_list = []
            for fsol in final_solutions:
                rmsd_list.append(cand[0].get_rmsdCA_with(fsol[0]))

            # We have a clone; add weight to the better solution
            if np.min(rmsd_list) < 6:
                final_solutions[np.argmin(rmsd_list)][3] += weight
                [final_solutions[np.argmin(rmsd_list)][7].append(x) for x in clustered_anchors]
                continue

            # We have a unique solution
            else:
                # Get RMSD with original
                rmsd = np.inf
                min_idx = np.inf
                for idx, ref_pdb in enumerate(ref_pdb_list):
                    rmsdca = hi_pdb.get_rmsdCA_with(ref_pdb)
                    if rmsd > rmsdca:
                        rmsd = rmsdca
                        min_idx = idx

                # Get CCC
                sub_grid, sx, sy, sz = hi_pdb.structure_to_density(res, dmap.voxsp)
                ccc = dmap.get_CCC_with_grid(sub_grid, sx, sy, sz)

                final_solutions.append([hi_pdb, corresp_anchors, repeatability, weight, ccc, rmsd, min_idx, clustered_anchors])

    # Sort results with new score
    for idx, sol in enumerate(final_solutions):
        hi_pdb, corresp_anchors, repeat, weight, ccc, rmsd, min_idx, clustered_anchors = sol
        super_score = repeat * weight * ccc
        sol.append(super_score)

    final_solutions = sorted(final_solutions, key=itemgetter(-1), reverse=True)

    # Save structures
    if len(ref_pdb_filelist):
        header = "|  # | Repeat | Weight |   mCC  |  RWmCC |  RMSD  | Ref"
        header_csv = ["ID", "Repeatability", "Weight", "mCC", "RWmCC", "RMSD", "Ref_PDB"]
        sep =    "-------------------------------------------------------"
    else:
        header = "|  # | Repeat | Weight |   mCC  |  RWmCC |"
        header_csv = ["ID", "Repeatability", "Weight", "mCC", "RWmCC"]
        sep =    "------------------------------------------"

    print("\n"+sep+"\n"+header+"\n"+sep)
    data_csv = []
    for idx, sol in enumerate(final_solutions):
        hi_pdb, corresp_anchors, repeat, weight, ccc, rmsd, min_idx, clustered_anchors, super_score = sol

        sol_fname = os.path.join(sol_path, "sol_%s_%i.pdb"%(hi_id, idx))
        hi_pdb.write_pdb(sol_fname)

        corresp_fname = os.path.join(sol_path, "corresp_anchors_%s_%i.pdb"%(hi_id, idx))
        save_coords_as_pdb(corresp_anchors, corresp_fname)

        save_oriented_anchors_as_pdb(clustered_anchors, sol_path, "%s_%i"%(hi_id, idx))

        if len(ref_pdb_filelist):
            sol_str =  "| %2i | %6.2f | %6i | %6.2f | %6.2f | %6.2f | %i"%(idx, repeat, weight, ccc, super_score, rmsd, min_idx)
            data_csv.append([idx, repeat, weight, ccc, super_score, rmsd, min_idx])
        else:
            sol_str =  "| %2i | %6.2f | %6i | %6.2f | %6.2f |"%(idx, repeat, weight, ccc, super_score)
            data_csv.append([idx, repeat, weight, ccc, super_score])
        print(sol_str)
    print(sep+"\n")
    if len(data_csv):
        pd.DataFrame(data_csv).to_csv(os.path.join(out_folder, "Solutions_refined_%s.csv"%hi_id), index=False, header=header_csv)


def _build_from_single(out_folder, chain_id, multiplicity, chain_dict, flist, final=False, max_models=10):
    # Distinguish between cases where we want subcomplexes and complexes made of a single component only
    if final:
        ind_sol_path = os.path.join(out_folder, "individual_solutions")
        comb_sol_path = os.path.join(out_folder, "assembly_models")
        if not os.path.exists(comb_sol_path):
            os.mkdir(comb_sol_path)
    else:
        ind_sol_path = os.path.join(out_folder, "individual_solutions")
        comb_sol_path = os.path.join(out_folder, "assembly_pieces")
        if not os.path.exists(comb_sol_path):
            os.mkdir(comb_sol_path)

    # Get solutions from folder
    solution_list = []
    for f in flist:
        for state in chain_dict[chain_id]:
            if f.startswith("sol_") and f.endswith(".pdb") and "_%s_%i"%(state, multiplicity) in f:
                # To sort properly filenames (otherwise, it goes 1, 10, 11, ..., 2, 20, ...)
                sol_id = int(f.split('_')[-1].replace(".pdb", ""))
                solution_list.append([sol_id, os.path.join(ind_sol_path, f)])
    solution_list = [x[1] for x in sorted(solution_list, key=itemgetter(0))]

    if multiplicity > len(solution_list):
        print("MaD> Not enough solutions to cover all copies for chain %s ! Maybe try increasing n_samples or reducing min_cc/wthresh ?"%chain_id)
        multiplicity = len(solution_list)

    if multiplicity > 1:
        # Get grids for overlap and prep for repeated structures
        map_list = []
        for idx, sol in enumerate(solution_list):
            grid, xi, yi, zi = PDB(sol).structure_to_density(1, 2) #resolution 1, voxelsp 2. Low res prevent biasing overlap
            map_list.append([idx, [grid, xi, yi, zi]])

        # Pre-compute overlaps between all pairs
        n_sol = len(solution_list)
        overlap_pairs = np.zeros((n_sol, n_sol))
        for c in combinations(map_list, 2):
            m1, m2 = c
            overlap = _get_overlap(m1[1], m2[1], 2)
            overlap_pairs[m1[0], m2[0]] = overlap

        # Show overlap table
        print("MaD> Pairwise overlaps between solutions of %s:"%chain_id)
        col_name = "      | "
        sol_id = []
        for x in range(n_sol):
            col_name = col_name + "%5s "%('.'.join([chain_id, str(x)]))
            sol_id.append('.'.join([chain_id, str(x)]))
        print()
        print(col_name)
        print("-"*len(col_name))
        row_cnt = 0
        for idx, s in enumerate(overlap_pairs):
            row = "%5s | "%('.'.join([chain_id, str(idx)]))
            row_cnt += 1
            for v in s:
                if v == 0.0:
                    row = row + "   0  "
                else:
                    row = row + "%.3f "%v
            print(row)
        print("-"*len(col_name))
        print()

        # For each candidate assembly, get total overlap
        print("MaD> Assembling %i copies of chain %s from %i solutions..."%(multiplicity, chain_id, n_sol))
        candidate_list = []
        cnt = 0
        for c in combinations(range(n_sol), multiplicity):
            X, Y = np.meshgrid(c, c)
            data_overlap = overlap_pairs[X.ravel(),Y.ravel()]
            max_overlap = np.max(data_overlap)
            std_overlap = np.std(data_overlap)
            sum_overlap = np.sum(data_overlap) / multiplicity
            candidate_list.append([c, sum_overlap, std_overlap, max_overlap])

        # Sort by increasing overlap sum
        candidate_list = sorted(candidate_list, key=itemgetter(3))

    # If single copy in complex, just save subcomplexes from original solutions.
    else:
        candidate_list = [[tuple([s_idx]), 0, 0, 0] for s_idx in range(len(solution_list))]

    # Save subcomplexes
    if not final:
        valid_candidates_list = []
        for s_idx, s in enumerate(candidate_list):
            sub_idx, score_sum, score_std, score_max = s
            if score_max > 0.1:
                continue

            ass_code = "_".join(["%s%i"%(chain_id, x) for x in sub_idx])

            comp_outname = os.path.join(comb_sol_path, "SubComplex%s_%i_%s.pdb"%(chain_id, s_idx, ass_code))
            write_complex_from_components([solution_list[idx] for idx in sub_idx], comp_outname)
            valid_candidates_list.append(comp_outname)
            #print("     %3i   AvgOverlap=%6.3f   MaxOverlap=%6.3f  StdOverlap=%6.3f SumOverlap=%6.3f"%(s_idx, score_sum, score_max, score_std, score_sum))

        if multiplicity > 1:
            print("MaD> Generated %i subcomplexes from component %s"%(len(valid_candidates_list), chain_id))
        return valid_candidates_list

    # Make final models if only a single component is given
    else:
        # Get assembly map
        low_map = [x for x in os.listdir(os.path.join(out_folder, "initial_files")) if x.endswith(".mrc")]
        if len(low_map):
            low_map = os.path.join(os.path.join(out_folder, "initial_files"), low_map[0])
        else:
            print("MaD> ERROR: could not find assembly map in %s; was it moved ?"%(os.path.join(out_folder, "initial_files")))
        low_name = os.path.split(low_map)[-1]

        # Compute CC / make PDBs
        # dmap = Dmap(low_map)
        print("MaD> Final models docked in map %s: "%low_name)
        print()
        comp_str = "Composition"
        n_spaces =  multiplicity * 4
        comp_str = comp_str + (n_spaces - len(comp_str)) * " "

        # Sort and display solutions
        header = "    # | %s  |   CC | Sum(O) | Std(O) | Max(O)"%comp_str
        header_csv = ["#", "Composition", "CC", "Sum(O)", "Std(O)", "Max(O)"]
        data_csv = []
        print(header)
        print("-" * len(header))
        for cnt, s in enumerate(candidate_list):
            sub_idx, score_sum, score_std, score_max = s

            if cnt >= max_models or (score_max > 0.1 and cnt):
                break
            cnt += 1
            ass_code = "_".join(["%s%i"%(chain_id, x) for x in sub_idx])
            ass_code_spaced = ass_code + (n_spaces - len(ass_code)) * " "

            comp_outname = os.path.join(comb_sol_path, "Model_%i_%s.pdb"%(cnt, ass_code))
            write_complex_from_components([solution_list[idx] for idx in sub_idx], comp_outname)
            #comp_pdb = PDB(comp_outname)
            #g, i1, i2, i3 = comp_pdb.structure_to_density(4, dmap.voxsp)
            #ccc = dmap.get_CCC_with_grid(g, i1, i2, i3)
            ccc = 0

            print("  %3i |  %s   disab %6.2f  %6.2f  %6.2f"%(cnt, ass_code_spaced, score_sum, score_std, score_max))
            data_csv.append([cnt, ass_code_spaced, ccc, score_sum, score_std, score_max])
        print("-" * len(header))
        pd.DataFrame(data_csv).to_csv(os.path.join(out_folder, "complex_ranking.csv"), index=False, header=header_csv)


def _build_models(out_folder, chain_id_list, sub_sol_list, max_models=10):
    print("MaD> Building assembly models from %i components..."%len(sub_sol_list))
    low_map = [x for x in os.listdir(os.path.join(out_folder, "initial_files")) if x.endswith(".mrc")]
    if len(low_map):
        low_map = os.path.join(os.path.join(out_folder, "initial_files"), low_map[0])
    else:
        print("MaD> ERROR: could not find assembly map in %s; was it moved ?"%(os.path.join(out_folder, "initial_files")))

    map_dic = {}

    sol_counter = 0
    sol_counter_list = []
    comp_sol_list = []
    comp_sol_file_list = []
    map_list_for_chain_id = []
    sol_chain_id_list = []
    for sol_list, chain_id in zip(sub_sol_list, chain_id_list):
        sol_counter_l = []
        for sol in sol_list:
            grid, xi, yi, zi = PDB(sol).structure_to_density(1, 2) #resolution 1, voxelsp 2. Low res prevent biasing overlap
            map_list_for_chain_id.append([grid, xi, yi, zi])
            comp_sol_list.append([chain_id, sol_counter])
            comp_sol_file_list.append(sol)
            sol_chain_id_list.append(chain_id)
            sol_counter_l.append(sol_counter)
            sol_counter += 1
        sol_counter_list.append(sol_counter_l)
        map_dic[chain_id] = map_list_for_chain_id

    # Pre-compute overlaps between all pairs
    n_sol = len(comp_sol_list)
    overlap_pairs = np.zeros((n_sol, n_sol))
    for c in combinations(comp_sol_list, 2):
        m1, m2 = c
        overlap = _get_overlap(map_dic[m1[0]][m1[1]], map_dic[m2[0]][m2[1]], 2)
        overlap_pairs[m1[1], m2[1]] = overlap

    col_name = "      | "
    sol_id = []
    for x in range(n_sol):
        col_name = col_name + "%5s "%('.'.join([comp_sol_list[x][0], str(comp_sol_list[x][1])]))
        sol_id.append('.'.join([comp_sol_list[x][0], str(comp_sol_list[x][1])]))
    print()
    print(col_name)
    print("-"*len(col_name))
    row_cnt = 0
    for s in overlap_pairs:
        row = "%5s | "%('.'.join([comp_sol_list[row_cnt][0], str(comp_sol_list[row_cnt][1])]))
        row_cnt += 1
        for v in s:
            if v == 0.0:
                row = row + "   0  "
            else:
                row = row + "%.3f "%v
        print(row)
    print("-"*len(col_name))
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
    ass_model_path = os.path.join(out_folder, "assembly_models")
    if not os.path.exists(ass_model_path):
        os.mkdir(ass_model_path)

    # Compute CC / make PDBs
    dmap = Dmap(low_map)
    low_name = os.path.split(low_map)[-1]

    print("MaD> Final models docked in map %s: "%low_name)
    print()
    comp_str = "Composition"
    n_spaces = len(chain_id_list) * 4
    comp_str = comp_str + (n_spaces - len(comp_str)) * " "

    # Sort and display solutions
    header = "    # | %s  |   CC | Sum(O) | Std(O) | Max(O)"%comp_str
    header_csv = ["#", "Composition", "CC", "Sum(O)", "Std(O)", "Max(O)"]
    dashes = "-" * len(header)
    print(header)
    print(dashes)
    data_csv = []
    for cnt, s in enumerate(solution_assemblies):
        sub_idx, score_sum, score_std, score_max = s

        if cnt >= max_models or (score_max > 0.1 and cnt):
            break
        cnt += 1
        ass_code = "_".join(["%s%i"%(sol_chain_id_list[x], x) for x in sub_idx])
        ass_code_spaced = ass_code + (n_spaces - len(ass_code)) * " "

        comp_outname = os.path.join(out_folder, "assembly_models", "Model_%i_%s.pdb"%(cnt, ass_code))
        write_complex_from_components([comp_sol_file_list[idx] for idx in sub_idx], comp_outname)
        comp_pdb = PDB(comp_outname)
        g, i1, i2, i3 = comp_pdb.structure_to_density(4, dmap.voxsp)
        ccc = dmap.get_CCC_with_grid(g, i1, i2, i3)
        # ccc = 0

        # print("  %3i |  %s   disab %6.2f   %6.2f   %6.2f"%(cnt, ass_code_spaced,  score_sum, score_std, score_max))
        print("  %3i |  %s    %6.2f %6.2f   %6.2f   %6.2f"%(cnt, ass_code_spaced, ccc, score_sum, score_std, score_max))
        data_csv.append([cnt, ass_code_spaced, ccc, score_sum, score_std, score_max])
    print("-" * len(header))
    pd.DataFrame(data_csv).to_csv(os.path.join(out_folder, "complex_ranking.csv"), index=False, header=header_csv)


def _get_overlap(g1, g2, voxsp, isovalue=1e-8):
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

####### I/O #######

def save_descriptors(df_list, outname):
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

def load_descriptors(input_name):
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

def save_matching(outname, match_results, map_anchors, comp_anchors):
    hf = h5py.File(outname, 'w')
    hf.create_dataset('match', data=match_results)
    hf.create_dataset('map_anchors', data=map_anchors)
    hf.create_dataset('comp_anchors', data=comp_anchors)
    hf.close()

def write_complex_from_components(components, outname):
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

def save_coords_as_pdb(coords, outname):
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

def save_oriented_anchors_as_pdb(anchor_list, sol_path, identifier):
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

        with open(os.path.join(sol_path, "anchor_%s_%s.pdb"%(target, identifier)), "w") as f:
            for x in pdb_data:
                line_model = "%-6s%5i  %-3s %3s%2s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s"
                line = line_model%("ATOM", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1, x[8], x[9])
                f.write(line+"\n")

        with open(os.path.join(sol_path, "anchor_ori_%s_%s.bld"%(target, identifier)), "w") as f:
            f.write(".color black\n")
            for x in bld_data_ori:
                f.write(".arrow %f %f %f %f %f %f 0.2 1.0 0.75\n"%(x[0], x[1], x[2], x[3], x[4], x[5]))

        if anchor_idx == 0:
            with open(os.path.join(sol_path, "anchor_cor_%s.bld"%(identifier)), "w") as f:
                f.write(".color black\n")
                for x in bld_data_cor:
                    f.write(".cylinder %f %f %f %f %f %f 0.1 \n"%(x[0], x[1], x[2], x[3], x[4], x[5]))



###### MISC ######

def get_repeatability(hi_coords, lo_coords, hi_pdb, hi_ori, voxelsp):

    # Load pdbs and get transformation for anchor points
    ori = PDB(hi_ori)
    mov = PDB(hi_pdb)

    # Apply transf to anchor points
    R, T = get_rototrans_SVD(mov.coords, ori.coords)

    moved_hi_coords = np.dot(hi_coords, R) + T

    # Compute repeatability and number of correspondences within threshold
    thresh = np.sqrt(2*voxelsp**2) * 1.1

    # Repeat w 1 neighbour
    kdtree = cKDTree(lo_coords)
    distances, indices = kdtree.query(moved_hi_coords, k=1)
    exp_n_corr = np.count_nonzero(distances < thresh)
    exp_rep = exp_n_corr / moved_hi_coords.shape[0]
    print(f"> (1neigh) Exp Repeatability={exp_rep:.2f}  nCorr={exp_n_corr}")

    # Repeat w 3 neighbour
    kdtree = cKDTree(lo_coords)
    distances, indices = kdtree.query(moved_hi_coords, k=3, distance_upper_bound=thresh)
    exp_n_corr = np.count_nonzero(distances < thresh)
    exp_rep = exp_n_corr / moved_hi_coords.shape[0]
    print(f"> (3neigh, with repetition) Exp Repeatability={exp_rep:.2f} nCorr={exp_n_corr}")

    return indices
