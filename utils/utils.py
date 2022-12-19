import os
import time
from glob import glob
import numpy as np
import SimpleITK as sitk
import torch


def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape)==2:
            pad_value = image[0,0]
        elif len(shape)==3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
    return res

def random_crop_3D_image_batched(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(img.shape) - 2), ("If you provide a list/tuple "
                    "as center crop make sure it has the same len as your data has dims (3d)")

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError("crop_size[0] must be smaller or equal to the images x dimension")

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

    if crop_size[2] < img.shape[4]:
        lb_z = np.random.randint(0, img.shape[4] - crop_size[2])
    elif crop_size[2] == img.shape[4]:
        lb_z = 0
    else:
        raise ValueError("crop_size[2] must be smaller or equal to the images z dimension")
    img_sub = img[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1], lb_z:lb_z+crop_size[2]]
    return img_sub

def random_data_argument(x, y):
    i = np.random.choice(range(10), 1)[0]
    if i == 0:
        return x*0.9, y
    elif i == 1:
        return x*1.1, y
    elif i in [2,3,4]:
        x = np.flip(x, axis=i).copy()
        y = np.flip(y, axis=i).copy()
        return x, y
    else:
        return x, y

def data_view(x, y, view_flip):
    view, bool_flip = view_flip.split('_')
    if y is None:
        y = np.ones([1,1,1,1,1])
    if view == 'axial':
        pass
    elif view == 'saggital':
        x = np.transpose(x, [0,1,3,4,2])
        y = np.transpose(y, [0,1,3,4,2])
    elif view == 'coronal':
        x = np.transpose(x, [0,1,4,2,3])
        y = np.transpose(y, [0,1,4,2,3])
    if bool_flip.lower() == 'flip':
        x = np.flip(x, 2).copy()
        y = np.flip(y, 2).copy()
    return x, y

def data_view_inverted(x, y, view_flip):
    'data_view的逆过程'
    view, bool_flip = view_flip.split('_')
    if y is None:
        y = np.ones([1,1,1,1,1])
    if bool_flip.lower() == 'flip':
        x = np.flip(x, 2).copy()
        y = np.flip(y, 2).copy()
    if view == 'axial':
        pass
    elif view == 'saggital':
        x = np.transpose(x, [0,1,4,2,3]) 
        y = np.transpose(y, [0,1,4,2,3])
    elif view == 'coronal':
        x = np.transpose(x, [0,1,3,4,2])
        y = np.transpose(y, [0,1,3,4,2])
    return x, y

def save_probmap(prob, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    prob_sitk = sitk.GetImageFromArray(prob.astype(np.float32))
    sitk.WriteImage(prob_sitk, save_path)

def save_image(image, save_path, save_original=False):
    '''Save iamge to 'save_path', which can be added '.nii.gz' automatically.
       And the iamge will be fell into the [155,240,240]
    '''
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not save_path.endswith('.nii.gz'):
        save_path += ".nii.gz"
    image_shape = np.array(image.shape)
    assert(len(image_shape) == 3), f"The iamge shape is {image_shape}. It must be 3-dimension"
    if np.sum([155,240,240] - image_shape) > 0:
        d0, h0, w0 = (0.5*([155,240,240] - image_shape)).astype(np.int)
        d,  h,  w  = image_shape
        label_like = np.zeros([155,240,240])
        label_like[d0:d0+d,h0:h0+h,w0:w0+w] = image
    else:
        label_like = image
    if save_original:
        image_sitk = sitk.GetImageFromArray(label_like)
    else:
        label_like = label_convert(label_like, [1,2,3], [2,1,4])
        image_sitk = sitk.GetImageFromArray(label_like.astype(np.uint8))
    sitk.WriteImage(image_sitk, save_path)

def label_convert(label, label_ls=[1,2,3], target_ls=[2,1,4]):
    "Convert the label value from 'label_ls' to 'target_ls'."
    label_like = np.zeros_like(label)
    for i,j in zip(label_ls, target_ls):
        label_like[label==i] = j
    return label_like

def label_split_to_channels(inputs, label_ls=[1,2,3], axis=1, remain_self=False):
    'get one hot map concatenated on dimension 1.'
    if remain_self:
        outs = [inputs]
    else:
        outs = []
    for i in label_ls:
        outs.append(np.array(inputs==i, dtype=np.int32))
    outs = np.concatenate(outs, axis=axis)
    return outs

def label_split_to_region(inputs, label_ls=[1,2,3], axis=1):
    'get one hot map concatenated on dimension 1.'
    outs = []
    for i in label_ls:
        outs.append(np.array(inputs >= i, dtype=np.uint8))
    outs = np.concatenate(outs, axis=axis)
    return outs

def datestr():
    'Get the real-time string.'
    tl = time.localtime()
    t_str = ('{t.tm_year:04}-{t.tm_mon:02}-{t.tm_mday:02} '
             '{t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}').format(t=tl)
    return t_str

def get_cross_idxs(txtname, i):
    'split the i-th cross folder from whole idxs'
    with open(txtname, 'r') as ftxt:
        lines = ftxt.readlines()
    val_idxs = lines[i].split()
    val_idxs = [int(x) for x in val_idxs]
    train_idxs = lines[:i]+lines[i+1:]
    train_idxs = ' '.join(train_idxs)
    train_idxs = train_idxs.split()
    train_idxs = [int(x) for x in train_idxs]
    return train_idxs, val_idxs

def chkdir(dir):
    if os.path.exists(dir):
        return dir
    else:
        os.makedirs(dir)
        return dir

def random_restore_pretrain(model, pretrain_dir, pre_folder_ls, model_xy, T=0.5):
    'Where the random map > T restore the saved params.'
    if pre_folder_ls is None:
        return model
    def get_latest(folder):
        paths = sorted(glob(f"{pretrain_dir}/{folder}/{model_xy}*"))
        return paths[-1]
    def get_pre_dict(net, folder):
        saved_state_dict = torch.load(get_latest(folder))
        new_params_dict = net.state_dict().copy()
        for name, param in new_params_dict.items():
            print(name, end='')
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            # if name in saved_state_dict and ('1.0' in name or '2.0' in name):

                random_dict = torch.where(torch.rand_like(param) > T, 
                            saved_state_dict[name].cpu(), new_params_dict[name])
                new_params_dict[name].copy_(random_dict)
                print('\t***   copy {}'.format(name))
            else:
                print()
        return new_params_dict
    model.net_0.load_state_dict(get_pre_dict(model.net_0, pre_folder_ls[0]))
    model.net_1.load_state_dict(get_pre_dict(model.net_1, pre_folder_ls[1]))
    model.net_2.load_state_dict(get_pre_dict(model.net_2, pre_folder_ls[2]))
    return model
