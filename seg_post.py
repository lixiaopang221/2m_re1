import os
import glob
import nibabel as nib
import numpy as np
from scipy import ndimage


def run():
    mask = 1
    T_et = 500
    seg_dirs = [
        r'F:\validation\brats_re\autocontext1\sv1_unet1',
        ]
    for seg_dir in seg_dirs:
        post(seg_dir, T_et, mask)

def clean_contour(img):
    # Smaller areas with lower prob are very likely to be false positives
    s_input = np.zeros_like(img)
    s_input[img > 0] = 1
    wt_mor = ndimage.binary_dilation(s_input, iterations=2).astype('int8')
    labels, num_featrues = ndimage.label(wt_mor)
    w_area = []
    for i in range(1, num_featrues+1):
        w_area.append(np.sum(s_input[labels == i]))
    if len(w_area) > 1:
        max_area = np.max(w_area)
        for i in range(len(w_area)):
            if w_area[i] < max_area / 3.0:
                img[labels == i + 1] = 0
    # img = ndimage.binary_fill_holes(img).astype(np.int8)
    return img

def post(seg_dir=None, T_et=200, mask=True):
    # seg_dir = r'F:\Xvalidation\gan1\single\t2_flair'
    # T_et = 200
    print(f'Directory: {seg_dir}')
    seg_dir_new = f'{os.path.dirname(seg_dir)}/post/{os.path.basename(seg_dir)}'
    if not os.path.exists(seg_dir_new):
        os.makedirs(seg_dir_new)
    paths = glob.glob(f'{seg_dir}/*.nii.gz')
    paths = sorted(paths)
    for path in paths:
        img = nib.load(path).get_data()
        img_name = os.path.basename(path)
        if mask:
            mask_x = get_mask(img_name)
            img = img * mask_x
        img = clean_contour(img)
        if np.sum(img==1)+np.sum(img==4) < 10:
            img[img==2] = 1
        if np.sum(img==4)<T_et:
            img[img==4] = 2
        img_nii = nib.AnalyzeImage(img.astype(np.uint8), None)
        nib.save(img_nii, f"{seg_dir_new}/{img_name}")
        print(img_name)
        
def get_mask(img_name):
        img_name = img_name.split('.')[0]
        img_dir = (r'E:\yan\dataset\BraTS\BRATS2018'+
                r'\MICCAI_BraTS_2018_Data_Validation')
        path = glob.glob(f'{img_dir}/{img_name}/*t1.nii.gz')[0]
        img = nib.load(path).get_data()
        mask = (img > 0).astype(np.int)
        return mask
        
if __name__=='__main__':        
    run()