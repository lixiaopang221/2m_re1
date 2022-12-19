import os
import numpy as np
import SimpleITK as sitk

# (T1,T1c) and (T2,FLEAR)
# (AP,VP)  and (NP,DP)

def ReadImage(path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


nii_idex='kidney_2020_67duailin'

t1_label_dir ='/data/ljf/kidney_tumor_nii_240/kidney_2020_56haoaiping/kidney_2020_56haoaiping_t1.nii.gz'
t1_brats_dir = '/data/ljf/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_t1.nii.gz'
t1_kidt = sitk.ReadImage(t1_label_dir)
t1_brats = sitk.ReadImage(t1_brats_dir)

seg_label_dir ='/data/ljf/kidney_tumor_nii_240/'+nii_idex+'/'+nii_idex+'_seg.nii.gz'
pred_brats_dir = '/home/gyguo/code/code/2m_re1/results/gan_22g_KidT/'+nii_idex+'.nii.gz'

seg_label = sitk.ReadImage(seg_label_dir)
pred_brats = sitk.ReadImage(pred_brats_dir)

print(1)
print(2)

