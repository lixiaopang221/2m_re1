
import os
import numpy as np
import SimpleITK as sitk

# (T1,T1c) and (T2,FLEAR)
# (AP,VP)  and (NP,DP)
def ReadImage(path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize() # 获取原图size
    originSpacing = itkimage.GetSpacing() # 获取原图spacing
    newSize = np.array(newSize, dtype='uint32')
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    resampler.SetReferenceImage(itkimage) # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage) # 得到重新采样后的图像
    return itkimgResampled

def resize_imges_and_save(img,save_dir,org_data):
    img.SetOrigin((0,0,0))
    img.SetSpacing(org_data.GetSpacing())
    img.SetDirection(org_data.GetDirection())
    re_img = resize_image_itk(img, org_data.GetSize(), resamplemethod=sitk.sitkNearestNeighbor)
    re_img.SetOrigin(img.GetOrigin())
    re_img.SetSpacing(img.GetSpacing())
    re_img.SetDirection(img.GetDirection())
    sitk.WriteImage(re_img, save_dir)

seg_label_dir ='/data/ljf/kidney_tumor_nii_2'
new_dir = '/data/ljf/kidney_tumor_nii_240_v2'
if os.path.exists(new_dir) is False:
    os.mkdir(new_dir)
t1_brats_dir = '/data/ljf/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_t1.nii.gz'
t1_brats = sitk.ReadImage(t1_brats_dir)
t2_brats_dir = '/data/ljf/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_t2.nii.gz'
t2_brats = sitk.ReadImage(t2_brats_dir)
t1ce_brats_dir = '/data/ljf/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_t1ce.nii.gz'
t1ce_brats = sitk.ReadImage(t1ce_brats_dir)
flair_brats_dir = '/data/ljf/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_flair.nii.gz'
flair_brats = sitk.ReadImage(flair_brats_dir)
seg_brats_dir = '/data/ljf/BraTS/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_seg.nii.gz'
seg_brats = sitk.ReadImage(seg_brats_dir)

for i, iterm in enumerate(os.listdir(seg_label_dir)):
    print(i)
    seg_dir = os.path.join(seg_label_dir, iterm, iterm + '_seg.nii')
    t1_dir = os.path.join(seg_label_dir, iterm, iterm + '_t1.nii.gz')
    t2_dir = os.path.join(seg_label_dir, iterm, iterm + '_t2.nii.gz')
    t1ce_dir = os.path.join(seg_label_dir, iterm, iterm + '_t1ce.nii.gz')
    flair_dir = os.path.join(seg_label_dir, iterm, iterm + '_flair.nii.gz')

    if os.path.exists( os.path.join(new_dir, iterm)) is False:
        os.mkdir(os.path.join(new_dir, iterm))

    t1 = sitk.ReadImage(t1_dir)
    t2 = sitk.ReadImage(t2_dir)
    t1ce = sitk.ReadImage(t1ce_dir)
    flair = sitk.ReadImage(flair_dir)
    seg = sitk.ReadImage(seg_dir)

    t1_new_dir = os.path.join(new_dir, iterm, iterm + '_t1.nii.gz')
    resize_imges_and_save(t1,t1_new_dir,t1_brats)


    t2_new_dir = os.path.join(new_dir, iterm, iterm + '_t2.nii.gz')
    resize_imges_and_save(t2,t2_new_dir,t2_brats)


    t1ce_new_dir = os.path.join(new_dir, iterm, iterm + '_t1ce.nii.gz')
    resize_imges_and_save(t1ce,t1ce_new_dir,t1ce_brats)


    flair_new_dir = os.path.join(new_dir, iterm, iterm + '_flair.nii.gz')
    resize_imges_and_save(flair,flair_new_dir,flair_brats)


    seg_new_dir = os.path.join(new_dir, iterm, iterm + '_seg.nii.gz')
    resize_imges_and_save(seg,seg_new_dir,seg_brats)