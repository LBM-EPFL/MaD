from mad import MaD

'''
Examples using the test data available on our lab's website:

# 6dbl, 5 A
    mad.add_map("experimental_data/6dbl/emd_7845_processed.mrc", 5)
    mad.add_subunit("experimental_data/6dbl/6dbl_subA.pdb", n_copies=1)
    mad.add_subunit("experimental_data/6dbl/6dbl_subB.pdb", n_copies=2)
    mad.add_subunit("experimental_data/6dbl/6dbl_subC.pdb", n_copies=1)
    mad.run()
    mad.build_assembly()

# 5up2, 6 A
    mad.add_map("experimental_data/5up2/emd_8581_processed.mrc", 6)
    mad.add_subunit("experimental_data/5up2/5up2_subA.pdb")
    mad.add_subunit("experimental_data/5up2/5up2_subB.pdb")
    mad.add_subunit("experimental_data/5up2/5up2_subC.pdb")
    mad.add_subunit("experimental_data/5up2/5up2_subD.pdb")
    mad.add_subunit("experimental_data/5up2/5up2_Fab.pdb")
    mad.run()
    mad.build_assembly()

# 5g4f, 7 A
    mad.add_map("experimental_data/5g4f/emd_3436_processed.mrc", 7)
    mad.add_subunit("experimental_data/5g4f/5g4f_subunit.pdb", n_copies=6)
    mad.run()

# 3j4k, 8 A
    mad.add_map("experimental_data/3j4k/emd_5751_processed.mrc", 8)
    mad.add_subunit("experimental_data/3j4k/3j4k_subunit.pdb", n_copies=5)
    mad.run()
    mad.build_assembly()

# 2p4n, 9 A
    mad.add_map("experimental_data/2p4n/emd_1340_processed.mrc", 9)
    mad.add_subunit("experimental_data/2p4n/2p4n_subA.pdb")
    mad.add_subunit("experimental_data/2p4n/2p4n_subB.pdb")
    mad.add_subunit("experimental_data/2p4n/2p4n_subC.pdb")
    mad.run(cc_threshold=0.5, n_samples=80)
    mad.build_assembly()

# 3j3u, 10 A
    mad.add_map("experimental_data/3j3u/emd_5609_processed.mrc", 10)
    mad.add_subunit("experimental_data/3j3u/3j3u_subunit.pdb", n_copies=6)
    mad.run(n_samples=100, cc_threshold=0.5)
    mad.build_assembly()
    
# 5kuh, 11.6 A
    mad.add_map("experimental_data/EMD-8290/8290_exp_map_processed.mrc", 11.6)
    mad.add_subunit("experimental_data/EMD-8290/5kuh_subA.pdb", n_copies=2)
    mad.add_subunit("experimental_data/EMD-8290/5kuh_subB.pdb", n_copies=2)
    mad.run(patch_size=24)
    mad.build_assembly()

# 4ckd, 13 A
    mad.add_map("experimental_data/EMD-2548/2548_exp_map_processed.mrc", 13)
    mad.add_subunit("experimental_data/EMD-2548/4CKD_subunit.pdb", n_copies=4)
    mad.run(n_samples=120, patch_size=12)
    mad.build_assembly()
'''

# Make a MaD instance
mad = MaD.MaD()

# Add map, specifiy its resolution
mad.add_map("experimental_data/5g4f/emd_3436_processed.mrc", 7)

# Add components
mad.add_subunit("experimental_data/5g4f/5g4f_subunit.pdb", n_copies=6)

# Get solutions per component
mad.run()

# Build assembly models from solutions
mad.build_assembly()
