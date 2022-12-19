import itertools
import math
import os
import subprocess as subp
import sys
import time
from glob import glob

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from setproctitle import setproctitle

from utils.dataset import BatchGenerator3D_random_sampling, load_dataset
from utils.model import Disc, M3DUNet
from utils.paths import DataPath

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Params:
    run_mode       = 'train' #---'train' or 'test'
    in_channels    = 2
    n_classes      = 2
    base_filters   = 16
    d_in_channels  = 2
    d_n_classes    = 1
    d_base_filters = 16
    steps          = 10000
    start_step     = 0
    batch_size     = 1
    learning_rate  = 1e-4
    momentum       = 0.9
    weight_decay   = 1e-5
    xlambda        = 10.0 #---10.0
    xlambda_idt    = 0 #---0.5
    print_freq     = 20
    save_freq      = 200
    modal_dict     = {'t1':0,'t1ce':1,'t2':2,'flair':3}  #---{'t1':0,'t1ce':1,'t2':2,'flair':3}
    model_path     = './model/gan/gan_22/' + '_'.join(modal_dict.keys())
    data_dir       = DataPath.preprocessed_training_data_folder
    train_idxs     = range(285)
    out_dir        = './gan_train'
    model_idx      = 5
    log_file_path  = './logs/GAN_train.txt'

def run():
    if len(sys.argv) >= 2:
        if sys.argv[1] in ['train','test']:
            Params.run_mode = sys.argv[1]
    if Params.run_mode == 'train':
        print('training......')
        train()
    elif Params.run_mode == 'test':
        print('testing......')
        test()

def train():
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(Params.model_path): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(Params.model_path)
    train_data = load_dataset(Params.train_idxs)
    train_data_generator = BatchGenerator3D_random_sampling(
                    train_data, Params.batch_size, Params.steps, None, convert_labels=True)
    gene_x2y = M3DUNet(Params.in_channels, Params.n_classes, Params.base_filters).to(device)
    gene_y2x = M3DUNet(Params.in_channels, Params.n_classes, Params.base_filters).to(device)
    disc_x   = Disc(Params.d_in_channels, Params.d_n_classes, Params.d_base_filters).to(device)
    disc_y   = Disc(Params.d_in_channels, Params.d_n_classes, Params.d_base_filters).to(device)

    d_opt = torch.optim.Adam(itertools.chain(
                    disc_x.parameters(), disc_y.parameters()),lr=Params.learning_rate)
    g_opt = torch.optim.Adam(itertools.chain(
                    gene_x2y.parameters(), gene_y2x.parameters()),lr=Params.learning_rate)
    criterion_l1  = torch.nn.L1Loss()
    criterion_gan = torch.nn.MSELoss()
    img_x2y_pool = ImagePool(10)
    img_y2x_pool = ImagePool(10)
    def set_grad(model_ls, rq_grad=False):
        if not hasattr(model_ls, '__iter__'):
            model_ls = [model_ls]
        for model in model_ls:
            for param in model.parameters():
                param.requres_grad = rq_grad

    def backward_g():
        if Params.xlambda_idt > 0:
            idt_x = gene_x2y(img_y)[1]
            loss_idt_x = Params.xlambda*Params.xlambda_idt*criterion_l1(idt_x, img_y)
            idt_y = gene_y2x(img_x)[1]
            loss_idt_y = Params.xlambda*Params.xlambda_idt*criterion_l1(idt_y, img_x)
        else:
            loss_idt_x, loss_idt_y = 0.0, 0.0
        fake_dy = disc_x(img_x2y)
        loss_gy_d = criterion_gan(fake_dy, torch.ones_like(fake_dy))
        fake_dx = disc_y(img_y2x)
        loss_gx_d = criterion_gan(fake_dx, torch.ones_like(fake_dx))
        loss_gx = Params.xlambda * criterion_l1(img_x2y2x, img_x)
        loss_gy = Params.xlambda * criterion_l1(img_y2x2y, img_y)
        gen_loss = loss_gx + loss_gy + loss_gx_d + loss_gy_d + loss_idt_x + loss_idt_y
        gen_loss.backward()
        return gen_loss

    def backward_d(net, img, fake):
        real_d = net(img)
        loss_real = criterion_gan(real_d, torch.ones_like(real_d))
        fake_d = net(fake.detach())
        loss_fake = criterion_gan(fake_d, torch.zeros_like(fake_d))
        loss_d = (loss_real + loss_fake) * 0.5
        loss_d.backward()
        return loss_d
    
    file_txt = open(Params.log_file_path, 'a')
    print(f'training between {Params.modal_dict}')
    print(f'training between {Params.modal_dict}', file=file_txt)
    for step in range(Params.start_step, Params.steps):
        data = next(train_data_generator)
        # modal_x, modal_y = Params.modal_dict.keys()
        modal_1, modal_2, modal_3, modal_4 = Params.modal_dict.values()
        modal_x = [modal_1, modal_2]
        modal_y = [modal_3, modal_4]
        img_x = data['data'][:,modal_x,...]
        img_y = data['data'][:,modal_y,...]
        img_x = torch.FloatTensor(img_x).to(device)
        img_y = torch.FloatTensor(img_y).to(device)
        img_x2y   = gene_x2y(img_x)[1]
        img_y2x   = gene_y2x(img_y)[1]
        img_x2y2x = gene_y2x(img_x2y)[1]
        img_y2x2y = gene_x2y(img_y2x)[1]

        set_grad([disc_x, disc_y], False)
        g_opt.zero_grad()
        gen_loss = backward_g()
        g_opt.step()
        set_grad([disc_x, disc_y], True)
        d_opt.zero_grad()
        loss_dx = backward_d(disc_x, img_y, img_x2y_pool(img_x2y))
        loss_dy = backward_d(disc_y, img_x, img_y2x_pool(img_y2x))
        d_opt.step()

        if step % Params.print_freq == 0:
            gen_loss_np = gen_loss.data.cpu().numpy()
            loss_dx_np = loss_dx.data.cpu().numpy()
            loss_dy_np = loss_dy.data.cpu().numpy()
            print(f"{datestr()} step:{step:<5} gen_loss:{gen_loss_np:.4f} "
                  f"loss_dx:{loss_dx_np:.4f} loss_dy:{loss_dy_np:.4f}")
            print(f"{datestr()} step:{step:<5} gen_loss:{gen_loss_np:.4f} "
                  f"loss_dx:{loss_dx_np:.4f} loss_dy:{loss_dy_np:.4f}", file=file_txt)
            if step % Params.save_freq == 0 and step>0:
                torch.save(gene_x2y.state_dict(), f'{Params.model_path}/x2y_{5*step//Params.steps}')
                torch.save(gene_y2x.state_dict(), f'{Params.model_path}/y2x_{5*step//Params.steps}')
                torch.save(disc_x.state_dict(), f'{Params.model_path}/dx_{5*step//Params.steps}')
                torch.save(disc_y.state_dict(), f'{Params.model_path}/dy_{5*step//Params.steps}')
    torch.save(gene_x2y.state_dict(), f'{Params.model_path}/x2y_5')
    torch.save(gene_y2x.state_dict(), f'{Params.model_path}/y2x_5')
    torch.save(disc_x.state_dict(), f'{Params.model_path}/dx_5')
    torch.save(disc_y.state_dict(), f'{Params.model_path}/dy_5')

def test():
    for x in glob(f'{Params.out_dir}/*.nii.gz'):
        os.remove(x)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_dataset(Params.train_idxs)
    train_data_generator = BatchGenerator3D_random_sampling(
                    train_data, Params.batch_size, Params.steps, None, convert_labels=True)
    gene_x2y = M3DUNet(Params.in_channels, Params.n_classes, Params.base_filters).to(device)
    gene_y2x = M3DUNet(Params.in_channels, Params.n_classes, Params.base_filters).to(device)
    disc_x   = Disc(Params.d_in_channels, Params.d_n_classes, Params.d_base_filters).to(device)#
    disc_y   = Disc(Params.d_in_channels, Params.d_n_classes, Params.d_base_filters).to(device)#
    gene_x2y.load_state_dict(torch.load(f"{Params.model_path}/x2y_{Params.model_idx}"))
    gene_y2x.load_state_dict(torch.load(f"{Params.model_path}/y2x_{Params.model_idx}"))
    disc_x.load_state_dict(torch.load(f"{Params.model_path}/dx_{Params.model_idx}"))#
    disc_y.load_state_dict(torch.load(f"{Params.model_path}/dy_{Params.model_idx}"))#
    criterion_l1  = torch.nn.L1Loss()
    criterion_gan = torch.nn.MSELoss()
    def loss_d(net, img, fake):
        real_d = net(img)
        loss_real = criterion_gan(real_d, torch.ones_like(real_d))
        fake_d = net(fake.detach())
        loss_fake = criterion_gan(fake_d, torch.zeros_like(fake_d))
        return loss_real.cpu(), loss_fake.cpu()

    for step in range(0, 5):
        data = next(train_data_generator)
        modal_1, modal_2, modal_3, modal_4 = Params.modal_dict.values()
        modal_x = [modal_1, modal_2]
        modal_y = [modal_3, modal_4]
        np_x = data['data'][:,modal_x,...]
        np_y = data['data'][:,modal_y,...]
        img_x = torch.FloatTensor(np_x).to(device)
        img_y = torch.FloatTensor(np_y).to(device)
        with torch.no_grad():
            img_x2y   = gene_x2y(img_x)[1]
            img_y2x   = gene_y2x(img_y)[1]
            img_x2y2x = gene_y2x(img_x2y)[1]
            img_y2x2y = gene_x2y(img_y2x)[1]
            loss_x_real, loss_x_fake = loss_d(disc_x, img_x, img_y2x)
            loss_y_real, loss_y_fake = loss_d(disc_y, img_y, img_x2y)
            print(loss_x_real.numpy(),loss_x_fake.numpy(),loss_y_real.numpy(),loss_y_fake.numpy(),)
        name = data['patient_names'][0]
        save_image(np_x[0,0,...], f'{Params.out_dir}/{name}_x', True)
        save_image(np_y[0,0,...], f'{Params.out_dir}/{name}_y', True)
        np_x2y = img_x2y.data.cpu().numpy()
        np_y2x = img_y2x.data.cpu().numpy()
        np_x2y2x = img_x2y2x.data.cpu().numpy()
        np_y2x2y = img_y2x2y.data.cpu().numpy()
        save_image(np_x2y[0,0,...], f'{Params.out_dir}/{name}_x2y', True)
        save_image(np_y2x[0,0,...], f'{Params.out_dir}/{name}_y2x', True)
        save_image(np_x2y2x[0,0,...], f'{Params.out_dir}/{name}_x2y2x', True)
        save_image(np_y2x2y[0,0,...], f'{Params.out_dir}/{name}_y2x2y', True)
        print(f'{name} have saved !')

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

def datestr():
    'Get the real-time string.'
    tl = time.localtime()
    t_str = ('{t.tm_year:04}-{t.tm_mon:02}-{t.tm_mday:02} '
             '{t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}').format(t=tl)
    return t_str

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def __call__(self, *xtp, **xdict):
        return self.query(*xtp, **xdict)

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = np.random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

if __name__ == '__main__':
    gpu_text = subp.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    memory_gpu = [int(x.split()[2]) for x in gpu_text.splitlines()]
    gpu_idx = str(np.argmax(memory_gpu))
    print('using gpu:', gpu_idx)
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_idx
    setproctitle('huanggh')
    t0 = datestr()
    run()
    t1 = datestr()
    print(f'start time:{t0}  ---  end time:{t1}')