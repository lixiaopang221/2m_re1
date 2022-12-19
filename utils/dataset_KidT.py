import itertools
import os
import pickle
from multiprocessing import Pool
from shutil import copyfile

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from .braintools import ReadImage
from .data_loader import DataLoaderBase
from .paths import DataPath_KidT as DataPath
from .utils import random_crop_3D_image_batched, reshape_by_padding_upper_coords


class PatchGenerator():
    'for test data'

    def __init__(self, data, patch_size=[128, 128, 128], batch_size=2, points=None):
        'data: [c,d,h,w]'
        self.data = data
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.data_shape = data.shape[1:]
        dd, hd, wd = self.data_shape
        dp, hp, wp = self.patch_size
        if points is None:
            self.points = itertools.product([0, dd - dp], [0, hd - hp], [0, wd - wp])
            self.points = list(self.points)
        else:
            self.points = points

    def get_batch_patch(self):
        patchs = []
        for p in self.points:
            z, y, x = p
            d, h, w = self.patch_size
            patch = self.data[:, z:z + d, y:y + h, x:x + w]
            patchs.append(patch)
        for idx in range(0, len(self.points), self.batch_size):
            batch_patch = patchs[idx:idx + self.batch_size]
            yield np.asarray(batch_patch)

    def patch_to_label(self, patch_ls, full_shape=None, bbox=None):
        '''patch_ls: num_patchs*[d,h,w].
           full_shape: If full_shape is None, return the label which shape is same as self.data.
                       Else, return the label which shape is full_shape
           bbox: [[z_min, z_max],[y_min, y_max],[x_min, x_max],], Not work when full_shape is None.
        '''
        patch_sum = np.sum(np.stack(patch_ls) > 0, axis=(1, 2, 3))
        idx_ls = patch_sum.argsort()
        data_like = np.zeros(self.data_shape)
        for idx in idx_ls:
            z, y, x = self.points[idx]
            d, h, w = self.patch_size
            data_like[z:z + d, y:y + h, x:x + w] = patch_ls[idx]
        if full_shape is None:
            return data_like.astype(np.uint8)
        label_like = np.zeros(full_shape)
        z, y, x = bbox[0][0], bbox[1][0], bbox[2][0]
        d, h, w = self.data_shape
        label_like[z:z + d, y:y + h, x:x + w] = data_like
        return label_like.astype(np.uint8)

    def patch_to_fullshape(self, patch_ls, full_shape=None, bbox=None):
        '''patch_ls: num_patchs*[c(4),d,h,w].
           full_shape: If full_shape is None, return the label which shape is same as self.data.
                       Else, return the label which shape is full_shape
           bbox: [[z_min, z_max],[y_min, y_max],[x_min, x_max],], do not work when full_shape is None.
        '''
        patch_sum = np.sum(np.stack(patch_ls)[:, 1, ...], axis=(1, 2, 3))
        idx_ls = patch_sum.argsort()
        data_like = np.zeros([4] + list(self.data_shape))
        for idx in idx_ls:
            z, y, x = self.points[idx]
            d, h, w = self.patch_size
            data_like[:, z:z + d, y:y + h, x:x + w] = patch_ls[idx]
        if full_shape is None:
            return data_like.astype(np.uint8)
        prob_like = np.zeros([4] + list(full_shape))
        z, y, x = bbox[0][0], bbox[1][0], bbox[2][0]
        d, h, w = self.data_shape
        prob_like[:, z:z + d, y:y + h, x:x + w] = data_like
        return prob_like

    def one_patch_to_fullshape(self, path, bbox, full_shape=[155, 240, 240]):
        label_like = np.zeros(full_shape)
        z, y, x = bbox[0][0], bbox[1][0], bbox[2][0]
        d, h, w = path.shape
        label_like[z:z + d, y:y + h, x:x + w] = path
        return label_like.astype(np.uint8)


def extract_brain_region(image, brain_mask, background=0):
    ''' find the boundary of the brain region,
        return the resized brain image and the index of the boundaries
    '''
    brain = np.where(brain_mask != background)
    min_z = int(np.min(brain[0]))
    max_z = int(np.max(brain[0])) + 1
    min_y = int(np.min(brain[1]))
    max_y = int(np.max(brain[1])) + 1
    min_x = int(np.min(brain[2]))
    max_x = int(np.max(brain[2])) + 1
    # ---resize image
    resizer = (slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))
    return image[resizer], [[min_z, max_z], [min_y, max_y], [min_x, max_x]]


def cut_off_values_upper_lower_percentile(
        image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    # print(cut_off_lower, cut_off_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    return res


def run(folder, out_folder, pat_id, name, return_if_no_seg=True, N4ITK=False):
    print(f"{pat_id:03} {name}")
    if N4ITK:
        t1_path = os.path.join(folder, f"{name}_t1_N4ITK_corrected.nii.gz")
        t1ce_path = os.path.join(folder, f"{name}_t1ce_N4ITK_corrected.nii.gz")
        t2_path = os.path.join(folder, f"{name}_t2_N4ITK_corrected.nii.gz")
        flair_path = os.path.join(folder, f"{name}_flair_N4ITK_corrected.nii.gz")
    if not N4ITK:
        t1_path = os.path.join(folder, f"{name}_t1.nii.gz")
        # print(t1_path)
        # exit()
        t1ce_path = os.path.join(folder, f"{name}_t1ce.nii.gz")
        t2_path = os.path.join(folder, f"{name}_t2.nii.gz")
        flair_path = os.path.join(folder, f"{name}_flair.nii.gz")
    seg_path = os.path.join(folder, f"{name}_seg.nii.gz")
    if not (os.path.isfile(t1_path) and os.path.isfile(t1ce_path)
            and os.path.isfile(t2_path) and os.path.isfile(flair_path)):
        print("T1, T1ce, T2 or Flair file does not exist")
        return
    if not os.path.isfile(seg_path):
        if return_if_no_seg:
            print("Seg file does not exist")
            return
    t1_img = sitk.ReadImage(t1_path)
    t1_nda = ReadImage(t1_path)
    t1ce_nda = ReadImage(t1ce_path)
    t2_nda = ReadImage(t2_path)
    flair_nda = ReadImage(flair_path)
    # print(t1_nda.shape, t1ce_nda.shape, t2_nda.shape, flair_nda.shape)
    try:
        seg_nda = ReadImage(seg_path)
    except RuntimeError:
        seg_nda = np.zeros(t1_nda.shape, dtype=np.float32)
    except IOError:
        seg_nda = np.zeros(t1_nda.shape, dtype=np.float32)

    original_shape = t1_nda.shape
    brain_mask = ((t1_nda != t1_nda[0, 0, 0]) & (t1ce_nda != t1ce_nda[0, 0, 0]) &
                  (t2_nda != t2_nda[0, 0, 0]) & (flair_nda != flair_nda[0, 0, 0]))
    resized_t1_nda, bbox = extract_brain_region(t1_nda, brain_mask, 0)
    resized_t1ce_nda, bbox1 = extract_brain_region(t1ce_nda, brain_mask, 0)
    resized_t2_nda, bbox2 = extract_brain_region(t2_nda, brain_mask, 0)
    resized_flair_nda, bbox3 = extract_brain_region(flair_nda, brain_mask, 0)
    resized_seg_nda, bbox4 = extract_brain_region(seg_nda, brain_mask, 0)
    assert bbox == bbox1 == bbox2 == bbox3 == bbox4
    assert (resized_t1_nda.shape == resized_t1ce_nda.shape
            == resized_t2_nda.shape == resized_flair_nda.shape)

    with open(os.path.join(out_folder, "%03.0d.pkl" % pat_id), 'wb') as fx:
        dp = {}
        dp['orig_shp'] = original_shape
        dp['bbox_z'] = bbox[0]
        dp['bbox_y'] = bbox[1]
        dp['bbox_x'] = bbox[2]
        dp['spacing'] = t1_img.GetSpacing()
        dp['direction'] = t1_img.GetDirection()
        dp['origin'] = t1_img.GetOrigin()
        pickle.dump(dp, fx)

    # setting the cut off threshold
    cut_off_threshold = 2.0

    def normalize(xi, mask=None):
        x = xi.copy()
        if mask is None:
            mask = x != 0
        x_temp = cut_off_values_upper_lower_percentile(
            x, mask, cut_off_threshold, 100.0 - cut_off_threshold)
        x[mask] = (x[mask] - x_temp[mask].mean()) / x_temp.std()
        # x_random = np.random.normal(0, 1, size = x.shape)
        # x[xi==0] = x_random[xi==0]
        # x[1-mask] = 0.00001 #------
        return x

    normalized_resized_t1_nda = normalize(resized_t1_nda)
    normalized_resized_t1ce_nda = normalize(resized_t1ce_nda)
    normalized_resized_t2_nda = normalize(resized_t2_nda)
    normalized_resized_flair_nda = normalize(resized_flair_nda)

    shp = resized_t1_nda.shape
    new_shape = np.array([128, 128, 128]) # 128 128 128
    pad_size = np.max(np.vstack((new_shape, np.array(shp))), 0)
    # print(pad_size)
    new_t1_nda = reshape_by_padding_upper_coords(normalized_resized_t1_nda, pad_size, 0)
    new_t1ce_nda = reshape_by_padding_upper_coords(normalized_resized_t1ce_nda, pad_size, 0)
    new_t2_nda = reshape_by_padding_upper_coords(normalized_resized_t2_nda, pad_size, 0)
    new_flair_nda = reshape_by_padding_upper_coords(normalized_resized_flair_nda, pad_size, 0)
    new_seg_nda = reshape_by_padding_upper_coords(resized_seg_nda, pad_size, 0)
    # print(new_t1_nda.shape, new_t1ce_nda.shape,
    #           new_t2_nda.shape, new_flair_nda.shape, new_seg_nda.shape)
    number_of_data = 5
    # print([number_of_data]+list(new_t1_nda.shape))

    all_data = np.zeros([number_of_data] + list(new_t1_nda.shape), dtype=np.float32)
    # print(all_data.shape)
    all_data[0] = new_t1_nda
    all_data[1] = new_t1ce_nda
    all_data[2] = new_t2_nda
    all_data[3] = new_flair_nda
    all_data[4] = new_seg_nda
    np.save(os.path.join(out_folder, "%03.0d" % pat_id), all_data)


def run_star(args):
    return run(*args)


def run_preprocessing_BraTS2018_training(
        training_data_location=DataPath.data_folder,
        training_data=DataPath.training_data,
        folder_out=DataPath.preprocessed_training_data_folder, N4ITK=False):
    if not os.path.isdir(folder_out):
        os.makedirs(folder_out)
    id_name_conversion = []
    patients = np.loadtxt(training_data, dtype=str)
    patients_path = [os.path.join(training_data_location, patient) for patient in patients]
    patients = [os.path.basename(patient) for patient in patients]
    ctr = len(patients)
    pool = Pool(5)
    pool.map(run_star, zip(patients_path, [folder_out] * len(patients),
                           range(ctr), patients, len(patients) * [True], len(patients) * [N4ITK]))
    pool.close()
    pool.join()

    #for i, j, p in zip(patients, range(ctr), patients_path):
    #    fx = 'HGG' if "HGG" in p else "LGG"
    #    id_name_conversion.append([i, j, fx])
    for i, j in zip(patients, range(ctr)):
        id_name_conversion.append([i, j, 'unknown'])

    id_name_conversion = np.vstack(id_name_conversion)
    np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")
    # copyfile(os.path.join(training_data_location, "survival_data.csv"),
    #                 os.path.join(folder_out, "survival_data.csv"))


def run_preprocessing_BraTS2018_validationOrTesting(
        original_data_location=DataPath.data_folder,
        raw_data=DataPath.validation_data,
        folder_out=DataPath.preprocessed_validation_data_folder, N4ITK=False):
    if not os.path.isdir(folder_out): os.makedirs(folder_out)
    id_name_conversion = []
    patients = np.loadtxt(raw_data, dtype=str)
    patients_path = [os.path.join(original_data_location, patient) for patient in patients]
    patients = [os.path.basename(patient) for patient in patients]
    ctr = len(patients)
    p = Pool(7)
    p.map(run_star, zip(patients_path, [folder_out] * len(patients), range(ctr),
                        patients, len(patients) * [False], len(patients) * [N4ITK]))
    p.close()
    p.join()

    for i, j in zip(patients, range(ctr)):
        id_name_conversion.append([i, j, 'unknown'])
    id_name_conversion = np.vstack(id_name_conversion)
    np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")
    # copyfile(os.path.join(original_data_location, "survival_data.csv"),
    #                 os.path.join(folder_out, "survival_data.csv"))


def load_dataset(pat_ids=range(285), folder=DataPath.preprocessed_training_data_folder):
    id_name_conversion = np.loadtxt(os.path.join(folder, "id_name_conversion.txt"), dtype="str")
    # print(id_name_conversion[0])
    # idxs = id_name_conversion[:, 1].astype(int)
    idx_dict = {int(k[1]): (k[0], k[2]) for k in id_name_conversion}
    # print(idxs)
    dataset = {}
    for pat in pat_ids:
        if os.path.isfile(os.path.join(folder, "%03.0d.npy" % pat)):
            dataset[pat] = {}
            dataset[pat]['idx'] = pat
            dataset[pat]['data'] = np.load(os.path.join(folder, "%03.0d.npy" % pat), mmap_mode='r')
            # dataset[pat]['name'] = id_name_conversion[np.where(idxs == pat)[0][0], 0]
            # dataset[pat]['type'] = id_name_conversion[np.where(idxs == pat)[0][0], 2]
            dataset[pat]['name'] = idx_dict[pat][0]
            dataset[pat]['type'] = idx_dict[pat][1]
            with open(os.path.join(folder, "%03.0d.pkl" % pat), 'rb') as fx:
                dp = pickle.load(fx)
            dataset[pat]['orig_shp'] = dp['orig_shp']
            dataset[pat]['bbox_z'] = dp['bbox_z']
            dataset[pat]['bbox_x'] = dp['bbox_x']
            dataset[pat]['bbox_y'] = dp['bbox_y']
            dataset[pat]['spacing'] = dp['spacing']
            dataset[pat]['direction'] = dp['direction']
            dataset[pat]['origin'] = dp['origin']
    return dataset


def convert_brats_seg(seg):
    new_seg = np.zeros(seg.shape, seg.dtype)
    new_seg[seg == 2] = 1
    new_seg[seg == 1] = 2
    # convert label 4 enhancing tumor to label 3
    new_seg[seg == 4] = 3
    return new_seg


def convert_to_brats_seg(seg):
    new_seg = np.zeros(seg.shape, seg.dtype)
    new_seg[seg == 1] = 2
    new_seg[seg == 2] = 1  # ------
    # convert label 3 back to label 4 enhancing tumor
    new_seg[seg == 3] = 4
    return new_seg


# Their code to generate 3D random batch for training
class BatchGenerator3D_random_sampling(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE, num_batches,
                 seed, patch_size=(128, 128, 128), convert_labels=False):
        self.convert_labels = convert_labels
        self._patch_size = patch_size
        DataLoaderBase.__init__(self, data, BATCH_SIZE, num_batches, seed)

    def generate_train_batch(self):
        ids = np.random.choice(list(self._data.keys()), self.BATCH_SIZE)
        data = np.zeros((self.BATCH_SIZE, 4, self._patch_size[0],
                         self._patch_size[1], self._patch_size[2]), dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 1, self._patch_size[0],
                        self._patch_size[1], self._patch_size[2]), dtype=np.int16)
        types = []
        patient_names = []
        identifiers = []
        ages = []
        survivals = []
        for j, i in enumerate(ids):
            types.append(self._data[i]['type'])
            patient_names.append(self._data[i]['name'])
            identifiers.append(self._data[i]['idx'])
            # construct a batch, not very efficient
            data_all = self._data[i]['data'][None]
            if np.any(np.array(data_all.shape[2:]) - np.array(self._patch_size) < 0):
                new_shp = np.max(np.vstack((np.array(data_all.shape[2:])[None],
                                            np.array(self._patch_size)[None])), 0)
                data_all = resize_image_by_padding_batched(data_all, new_shp, 0)
            data_all = random_crop_3D_image_batched(data_all, self._patch_size)
            data[j, :] = data_all[0, :4]
            if self.convert_labels:
                seg[j, 0] = convert_brats_seg(data_all[0, 4])
            else:
                seg[j, 0] = data_all[0, 4]
            if 'survival' in self._data[i].keys():
                survivals.append(self._data[i]['survival'])
            else:
                survivals.append(np.nan)
            if 'age' in self._data[i].keys():
                ages.append(self._data[i]['age'])
            else:
                ages.append(np.nan)
        data_dict = {"data": data, "seg": seg, "idx": ids, "grades": types, "identifiers": identifiers,
                     "patient_names": patient_names, "survival": survivals, "age": ages}
        return data_dict


class TrainDataGenerator():
    def __init__(self, data, batch_size, patch_size=[128, 128, 128], convert_labels=True):
        self.data = data
        self.idxs = list(data.keys())
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.convert_labels = convert_labels

    def shuffle_idxs(self):
        np.random.shuffle(self.idxs)

    def batch_generator(self):
        for i in range(0, len(self.idxs), self.batch_size):
            idx_batch = self.idxs[i:i + self.batch_size]
            data, seg = [], []
            idxs, names = [], []
            for idx in idx_batch:
                idxs.append(self.data[idx]['idx'])
                names.append(self.data[idx]['name'])
                data_i = self.data[idx]['data'][None]
                if np.any(np.array(data_i.shape[2:]) - np.array(self.patch_size) < 0):
                    new_shp = np.max(np.vstack(
                        (np.array(data_i.shape[2:])[None],
                         np.array(self.patch_size)[None])), 0)
                    data_i = resize_image_by_padding_batched(data_i, new_shp, 0)
                data_i = random_crop_3D_image_batched(data_i, self.patch_size)
                data.append(data_i[0, :4])
                if self.convert_labels:
                    seg.append(convert_brats_seg(data_i[0, 4:5]))
                else:
                    seg.append(data_i[0, 4:5])
            data = np.stack(data, axis=0)
            seg = np.stack(seg, axis=0)
            data_dict = dict(data=data, seg=seg, names=names, idxs=idxs)
            yield data_dict
