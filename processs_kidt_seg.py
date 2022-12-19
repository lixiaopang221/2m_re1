import os
import numpy as np

# (T1,T1c) and (T2,FLEAR)
# (AP,VP)  and (NP,DP)
seg_label_dir ='/data/ljf/kidney_tumor_nii_2'
for i, iterm in enumerate(os.listdir(seg_label_dir)):

    seg_dir_orname  = os.path.join(seg_label_dir,iterm,iterm+'_AP_seg.nii.gz')
    seg_dir_newname = os.path.join(seg_label_dir, iterm, iterm + '_seg.nii')
    if os.path.exists(seg_dir_orname):
        os.rename(seg_dir_orname,seg_dir_newname)

print(i)