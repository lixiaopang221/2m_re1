import logging
import os
import subprocess as subp
import sys
from glob import glob

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from setproctitle import setproctitle

from utils.dataset import (BatchGenerator3D_random_sampling, PatchGenerator,
                           load_dataset)
from utils.loss import DiceLoss, DiceNp
from utils import model2 as model_factory
from utils.paths import DataPath
from utils.utils import (chkdir, datestr, label_convert,
                         label_split_to_channels, label_split_to_region,
                         random_data_argument, random_restore_pretrain,
                         save_image, save_probmap)


class Params:
    run_mode      = 'train'  #---'train' or 'test' or testprob or fusion
    in_channels   = 4
    num_classes   = 4
    base_n_filter = 20
    batch_size    = 2
    patch_size    = [128,128,128]
    modal_dict    = {'t1':0,'t1ce':1,'t2':2,'flair':3}  #---{'t1':0,'t1ce':1,'t2':2,'flair':3}

    #---Training setting
    steps         = 20000
    start_step    = 0
    learning_rate = 1e-4
    momentum      = 0.9
    weight_decay  = 1e-5
    model_w_init  = None #---'normal_', 'uniform_ ...', 'kaiming_normal_'
    pre_model     = None #---'./model/gan_22g_train_on17/model_4'
    pretrain_dir  = './model/gan/gan_22/'
    pre_folder_ls = None #--- ["t1_t1ce_t2_flair"]
    model_xy      = 'x2y'
    model_yx      = 'y2x'
    print_freq    = 20
    save_freq     = 200
    model_path    = './model/unet_1718'
    txt_postfix   = 'unet_1718' + '_' + '_'.join(modal_dict.keys())
    data_dir      = DataPath.preprocessed_training_data_folder
    data_argument = False

    #---Testing setting
    saved_model    = './model/unet_1718/model_4'
    save_prob      = False
    test_save_dir  = './results/unet_1718'
    test_data_idxs = list(range(66))
    test_data_path = DataPath.preprocessed_validation_data_folder

def run():
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) >= 2:
        if sys.argv[1] in ['train','test']:
            Params.run_mode = sys.argv[1]
    if Params.run_mode == 'train':
        train(device)
    elif Params.run_mode == 'test':
        test(device)
    elif Params.run_mode == 'testprob':
        testprob(device)
    elif Params.run_mode == 'fusion':
        fusion()
    # Training setting
    # test_cross(device, int(cross_idx))

def train(device):
    print(f'model saving path:{Params.model_path}')
    log_dir = chkdir("logs")
    log_file = os.path.join(log_dir, f"train_{Params.txt_postfix}.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)

    #---Loading the model
    model = model_factory.M3DUNet(Params.in_channels, Params.num_classes, Params.base_n_filter)
    if Params.model_w_init is not None:
        model = model_init(model)
    elif Params.pre_model is not None:
        if os.path.isfile(Params.pre_model):
            model.load_state_dict(torch.load(Params.pre_model))
            print(f'Loaded the model: {Params.pre_model}')
    else:
        model = restore_pretrain(model)
    model.train()
    model.to(device)
    train_idxs = list(range(285))
    train_data = load_dataset(train_idxs)
    train_data_17 = load_dataset(train_idxs, "../../data/preprpcess/BraTS_2017_train")
    for x in train_idxs:
        train_data_17[x+285] = train_data_17.pop(x)
    train_data.update(train_data_17)
    train_data_generator = BatchGenerator3D_random_sampling(
                    train_data, Params.batch_size, Params.steps, None, convert_labels=True)
    criterion = DiceLoss()

    #---create optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                    lr=Params.learning_rate, weight_decay=Params.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=Params.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.8)

    #---Training loop
    file_txt = open(log_file, 'a')
    print(f"Training on {len(train_idxs)} samples. Training {Params.steps} steps.")
    print(f"Training on {len(train_idxs)} samples. "
          f"Training {Params.steps} steps.", file=file_txt)
    for step in range(Params.start_step, Params.steps):
        data    = next(train_data_generator)
        # targets = label_split_to_channels(data['seg'], label_ls=[1,2,3])
        data_x = data['data']
        data_y = data['seg']
        data_y = label_split_to_channels(data_y, label_ls=[1,2,3])
        if Params.data_argument:
            data_x, data_y = random_data_argument(data_x, data_y)
        targets = torch.FloatTensor(data_y).to(device)

        inputs  = torch.FloatTensor(data_x).to(device)
        xf, *_ = model(inputs)
        # loss_0 = criterion(x0[:,1:,...], targets)
        loss = criterion(xf[:,1:], targets)
        optimizer.zero_grad()
        # scheduler.step()
        loss.backward()
        optimizer.step()
        if step % Params.print_freq == 0:
            loss_np = loss.data.cpu().numpy().copy()
            print(f'{datestr()} step:{step:<5} loss:{loss_np:.4f}')
            print(f'{datestr()} step:{step:<5} loss:{loss_np:.4f}', file=file_txt)
            
            if step!=0 and step % Params.save_freq == 0:
                if not os.path.exists(f"{Params.model_path}"):
                    os.makedirs(f"{Params.model_path}")
                torch.save(model.state_dict(), 
                            f'{Params.model_path}/model_{5*step//Params.steps}')
                xf_np = xf.data.cpu().numpy().copy()
                output_img = np.argmax(xf_np, axis=1)[0]
                tergets_np = data['seg'][0,0,...]
                save_image(output_img, f"./train_pred/{step:04}_o.nii.gz")
                save_image(tergets_np, f"./train_pred/{step:04}_l.nii.gz")
    print(f'The last model is saved: {Params.model_path}/model_4')
    file_txt.close()

def test(device):
    #---Loading the model
    print(f"The model path is {Params.saved_model}")    #------
    print(f"Valuing {len(Params.test_data_idxs)} samples.")
    model = model_factory.M3DUNet(Params.in_channels, Params.num_classes, Params.base_n_filter)
    model.load_state_dict(torch.load(Params.saved_model))
    model.eval()
    model.to(device)
    test_data = load_dataset(Params.test_data_idxs, folder=Params.test_data_path)
    for idx in Params.test_data_idxs:
        data = test_data[idx]
        print(f"valuing...... {data['name']}")
        pgen = PatchGenerator(data['data'][:4], Params.patch_size, Params.batch_size)
        output_ls = []
        for batch_patch in pgen.get_batch_patch():
            inputs  = torch.FloatTensor(batch_patch).to(device)
            with torch.no_grad():
                xf, *_ = model(inputs)
            xf = xf.data.cpu().numpy().copy()
            if not Params.save_prob:
                xf = np.argmax(xf, axis=1)
            output_ls.extend(xf)
        bbox = [data['bbox_z'], data['bbox_y'], data['bbox_x']]
        if Params.save_prob:
            prob = pgen.patch_to_fullshape(output_ls, data['orig_shp'], bbox)
            save_probmap(prob, f"{Params.test_save_dir}/{data['name']}.nii.gz")
        else:
            label = pgen.patch_to_label(output_ls, data['orig_shp'], bbox)
            save_image(label, f"{Params.test_save_dir}/{data['name']}.nii.gz")

def fusion():
    subdir_ls = [
        './results/gan_22g_and_pre17/gan_22g',
        './results/gan_22g_and_pre17/gan_22g_pre17_ep3'
    ]
    samples = os.listdir(subdir_ls[0])
    samples = [x for x in samples if x.endswith('.nii.gz')]
    for sample in samples:
        print(f'fusing {sample}')
        prob_ls = []
        for subdir in subdir_ls:
            sample_path = f'{subdir}/{sample}'
            prob_sitk = sitk.ReadImage(sample_path)
            prob = sitk.GetArrayFromImage(prob_sitk)
            prob_ls.append(prob)
        prob_mean = np.mean(prob_ls, 0)
        predict = np.argmax(prob_mean, 0)
        # predict = label_convert(predict, [1,2,3], [2,1,4])
        save_image(predict, f'{Params.test_save_dir}/fusion/{sample}')

def restore_pretrain(model):
    if Params.pre_folder_ls is None:
        return model
    def get_latest(mode):
        paths = sorted(glob(f"{Params.pretrain_dir}/{Params.pre_folder_ls[0]}/{mode}*"))
        return paths[-1]
    def get_pre_dict(net, mode):
        print(get_latest(mode))
        saved_state_dict = torch.load(get_latest(mode))
        new_params_dict = net.state_dict().copy()
        for name, param in new_params_dict.items():
            print(name, end='')
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            # if name in saved_state_dict and ('1.0' in name or '2.0' in name):
                new_params_dict[name].copy_(saved_state_dict[name])
                print('\t***   copy {}'.format(name))
            else:
                print()
        return new_params_dict
    model.net_0.load_state_dict(get_pre_dict(model.net_0, Params.model_xy))
    model.net_1.load_state_dict(get_pre_dict(model.net_1, Params.model_yx))
    return model

def model_init(model):
    init_mode = getattr(nn.init, Params.model_w_init)
    def weight_init(m):
        calss_name = m.__class__.__name__
        if calss_name.find('Conv') != -1:
            # nn.init.normal_(m.weight.data)
            init_mode(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
    model.apply(weight_init)
    print(f'Apply the initialization: {Params.model_w_init}')
    return model

if __name__ == '__main__':
    setproctitle('huanggh')
    gpu_text = subp.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    gpu_idx = np.argmax([int(x.split()[2]) for x in gpu_text.splitlines()])
    print('using......:', gpu_idx)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    run()
