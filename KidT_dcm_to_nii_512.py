import dicom2nifti
import os
import shutil
import SimpleITK as sitk
# from dcm to nii
# (T1,T1c) and (T2,FLEAR)
# (AP,VP)  and (NP,DP)

# AP:t1 VP:t1ce NP:t2 DP:flair

dicom_root = "/data/ljf/kidney_tumor"

converted_path = "/data/ljf/kidney_tumor_nii_2"
if os.path.exists(converted_path) is False:
    os.mkdir(converted_path)


def dcm2nii(dcms_path, nii_path):

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()

    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z

    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)

for i,item in enumerate(os.listdir(dicom_root)):
    for j,subitem in enumerate(os.listdir(os.path.join(dicom_root,item))):
        for kk, subsubitem in enumerate(os.listdir(os.path.join(dicom_root, item,subitem))):
            dcm_dir=os.path.join(dicom_root, item, subitem,subsubitem)
            mods_in_dir=os.listdir(dcm_dir)
            sub_set=['NP', 'DP', 'VP', 'AP', 'DP_Merge.nii', 'NP_Merge.nii', 'VP_Merge.nii', 'AP_Merge.nii']
            if set(sub_set).issubset(set(mods_in_dir)):
                print(dcm_dir)
                print(kk)
                nii_name = item+'_'+subsubitem
                nii_dir = os.path.join(converted_path,nii_name)

                if os.path.exists(nii_dir) is False:
                    os.mkdir(nii_dir)


                NP_nii_dir = os.path.join(nii_dir, nii_name + '_t2.nii.gz')

                DP_nii_dir = os.path.join(nii_dir, nii_name + '_flair.nii.gz')

                VP_nii_dir = os.path.join(nii_dir, nii_name + '_t1ce.nii.gz')

                AP_nii_dir = os.path.join(nii_dir, nii_name + '_t1.nii.gz')

                NP_dcm_dir = os.path.join(dcm_dir, 'NP')
                DP_dcm_dir = os.path.join(dcm_dir, 'DP')
                VP_dcm_dir = os.path.join(dcm_dir, 'VP')
                AP_dcm_dir = os.path.join(dcm_dir, 'AP')

                dcm2nii(NP_dcm_dir, NP_nii_dir)

                dcm2nii(DP_dcm_dir, DP_nii_dir)

                dcm2nii(VP_dcm_dir, VP_nii_dir)

                dcm2nii(AP_dcm_dir, AP_nii_dir)

                NP_Merge_dir = os.path.join(dcm_dir, 'NP_Merge.nii')
                DP_Merge_dir = os.path.join(dcm_dir, 'DP_Merge.nii')
                VP_Merge_dir = os.path.join(dcm_dir, 'VP_Merge.nii')
                AP_Merge_dir = os.path.join(dcm_dir, 'AP_Merge.nii')

                #shutil.copy(NP_Merge_dir, os.path.join(nii_dir, nii_name + '_NP_seg.nii.gz'))
                #shutil.copy(DP_Merge_dir, os.path.join(nii_dir, nii_name + '_DP_seg.nii.gz'))
                #shutil.copy(VP_Merge_dir, os.path.join(nii_dir, nii_name + '_VP_seg.nii.gz'))
                shutil.copy(AP_Merge_dir, os.path.join(nii_dir, nii_name + '_seg.nii'))

                print("end")





