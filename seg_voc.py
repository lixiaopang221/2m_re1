import os
import subprocess as subp
from glob import glob
import sys
import numpy as np
import torch
import cv2
from utils.model2 import M2d
from utils.utils import chkdir, datestr
from utils.loss import DiceLoss, CrossEntropy2d, cross_entropy2d


class Params():
    def read_txt(txt_name):
        with open(txt_name) as f_txt:
            data_ls = f_txt.read().splitlines()
        return data_ls

    run_mode      = 'test' #---test or train
    in_channels   = 3
    num_classes   = 21
    base_n_filter = 16
    image_size    = 256 # 128, 256, 384, 512
    epochs        = 100
    batch_size    = 20
    learning_rate = 0.001
    print_freq    = 20
    save_freq     = 200
    model_dir     = chkdir('./model/voc')
    txt_log       = './logs/voc_re_train.txt'
    fine_tune     = True
    

    data_ls = read_txt(
              '../../data/voc2012/ImageSets/SegmentationAug/train_aug.txt')
    img_ls  = [f'../../data/voc2012/JPEGImages/{x}.jpg' for x in data_ls]
    seg_ls  = [f'../../data/voc2012/SegmentationClassAug/{x}.png' for x in data_ls]
    
    saved_model   = './model/voc_re/model_4'
    test_save_dir = chkdir('./results/voc')
    test_data_ls  = read_txt('../../data/voc2012/ImageSets/Segmentation/val.txt')
    img_test_ls   = [f'../../data/voc2012/JPEGImages/{x}.jpg' for x in test_data_ls]
    seg_test_ls   = [f'../../data/voc2012/SegmentationClass/{x}.png' for x in test_data_ls]

def train(device):
    model = M2d(Params.in_channels, Params.num_classes, Params.base_n_filter)
    if Params.fine_tune:
        model.load_state_dict(torch.load(Params.saved_model))
    model.train()
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    # loss_func = DiceLoss(reduce_dim=[2,3])
    optimizer = torch.optim.Adam(model.parameters(), lr=Params.learning_rate)
    file_txt = open(Params.txt_log, 'a') 
    loader = DataLoad(Params.img_ls, Params.seg_ls, seg_aug=True, size_op='crop')
    print(f'number of samples:{len(Params.img_ls)}, fine_tune:{Params.fine_tune}')
    epochs = Params.epochs
    step = 0
    for epoch in range(epochs):
        loader.shuffle_idx()
        loss_ls = []
        for x_batch, y_batch in loader.get_batch(Params.batch_size, Params.image_size):
            inputs = torch.FloatTensor(x_batch.transpose([0,3,1,2])).to(device)
            # target = torch.FloatTensor(y_batch.transpose([0,3,1,2])).to(device)
            target = torch.FloatTensor(y_batch).to(device).long()
            output = model(inputs)[1]
            # loss = loss_func(output[:,1:,...], target[:,1:,...])
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            loss_np = loss.data.cpu().numpy()
            loss_ls.append(loss_np)
            if step % Params.print_freq == 0:
                print(f'{datestr()} epoch:{epoch}/{epochs} step:{step} loss:{loss_np:.6f}')
                print(f'{datestr()} epoch:{epoch}/{epochs} '
                            f'step:{step} loss:{loss_np:.4f}', file=file_txt)
            if step % Params.save_freq == 0:
                torch.save(model.state_dict(), 
                            f'{Params.model_dir}/model_{5*epoch//epochs}')
        loss_mean = np.mean(loss_ls)
        print(f'{datestr()} epoch:{epoch}/{epochs} loss_mean:{loss_mean:.4f}')
        print(f'{datestr()} epoch:{epoch}/{epochs} loss_mean:{loss_mean:.4f}', file=file_txt)
    torch.save(model.state_dict(), f'{Params.model_dir}/model_4')

def test(device):
    model = M2d(Params.in_channels, Params.num_classes, Params.base_n_filter)
    model.load_state_dict(torch.load(Params.saved_model))
    model.eval()
    model.to(device)
    loader = DataLoad(Params.img_test_ls, Params.seg_test_ls)
    print(f'Test the model:{Params.saved_model}')
    for idx in loader.idxs:
        x, y = loader.load_test_data(idx)
        x = x.transpose([2,0,1])
        inputs = torch.FloatTensor(x[None,...]).to(device)
        with torch.no_grad():
            output = model(inputs)[0]
            # output = torch.sigmoid(output)
        label = np.argmax(output.data.cpu().numpy(), axis=1)
        label_color = loader.colorize(label[0])
        img_name = os.path.basename(loader.x_file_list[idx])
        print(f'testing... {img_name}')
        cv2.imwrite(f'{Params.test_save_dir}/{img_name}', label_color)


def run():
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) >= 2:
        if sys.argv[1] in ['train','test']:
            Params.run_mode = sys.argv[1]
    if Params.run_mode == 'train':
        train(device)
    elif Params.run_mode == 'test':
        test(device)

class DataLoad():
    def __init__(self, x_file_list, y_file_list, seg_aug=False, size_op='resize'):
        self.colormap = [[ 0, 0,  0], [128, 0,  0], [ 0,128,  0], [128,128,  0],
                         [ 0, 0,128], [128, 0,128], [ 0,128,128], [128,128,128],
                         [64, 0,  0], [192, 0,  0], [64,128,  0], [192,128,  0],
                         [64, 0,128], [192, 0,128], [64,128,128], [192,128,128],
                         [ 0,64,  0], [128,64,  0], [ 0,192,  0], [128,192,  0],
                         [ 0,64,128], ]
        self.file_length = len(x_file_list) 
        self.idxs = np.arange(self.file_length)
        self.x_file_list = x_file_list
        self.y_file_list = y_file_list
        self.seg_aug = seg_aug
        self.size_op = size_op

    def random_crop(self, image_in, label_in, size):
        'size:[h,w] (list)'
        image = image_in.copy()
        label = label_in.copy()
        size = size if isinstance(size, list) else [size, size]
        while image.shape[0]-size[0] <= 0:
            image = np.concatenate([image]*2, axis=0)
            label = np.concatenate([label]*2, axis=0)
        while image.shape[1]-size[1] <= 0:
            image = np.concatenate([image]*2, axis=1)
            label = np.concatenate([label]*2, axis=1)
        idx_h = np.random.randint(image.shape[0]-size[0])
        idx_w = np.random.randint(image.shape[1]-size[1])
        image_crop = image[idx_h:idx_h+size[0],idx_w:idx_w+size[1]]
        label_crop = label[idx_h:idx_h+size[0],idx_w:idx_w+size[1]]
        return image_crop, label_crop

    def format_label(self, label):
        h, w = label.shape[:2]
        cmap = np.array(self.colormap)
        cmap = np.stack([cmap]*w, axis=0)
        cmap = np.stack([cmap]*h, axis=0)
        label_c = np.stack([label]*len(self.colormap), axis=-2)
        label_c = np.where(label_c==cmap, 1, 0)
        label_c = np.sum(label_c, axis=-1)
        label_c = np.where(label_c==3, 1, 0)
        return label_c

    def shuffle_idx(self):
        np.random.shuffle(self.idxs)

    def load_data(self, idx, size): 
        x_line = self.x_file_list[idx] 
        y_line = self.y_file_list[idx] 
        x_image = cv2.imread(x_line, 1) 
        if self.seg_aug:
            y_image = cv2.imread(y_line, 0) 
            # y_image_ls = []
            # for i in range(21):
            #     y_image_ls.append(np.array(y_image==i, dtype=np.uint8))
            # y_image = np.stack(y_image_ls, axis=-1)
        else:
            y_image = cv2.imread(y_line, 1) 
            y_image = cv2.cvtColor(y_image, cv2.COLOR_BGR2RGB)
            y_image = self.format_label(y_image)
            # y_image = np.argmax(y_image, axis=2)
        x_image = x_image/127.5-1. #归一化x域的图片
        if self.size_op == 'resize':
            x_image = cv2.resize(x_image, (size,size))
            y_image = cv2.resize(y_image, (size,size), interpolation=cv2.INTER_NEAREST)
        else:
            x_image, y_image = self.random_crop(x_image, y_image, size)
        return x_image, y_image 

    def get_batch(self, batch_size, img_size):
        for i in range(0, self.idxs.size, batch_size):
            x_batch, y_batch = [], []
            ib_ls = self.idxs[i:i+batch_size]
            for ib in ib_ls:
                x, y = self.load_data(ib, img_size)
                x_batch.append(x)
                y_batch.append(y)
            yield np.stack(x_batch), np.stack(y_batch)

    def crop_x16(self, img):
        h, w = img.shape[:2]
        h = h//16*16
        w = w//16*16
        img = img[:h,:w]
        return img

    def load_test_data(self, idx): 
        x_line = self.x_file_list[idx] 
        y_line = self.y_file_list[idx] 
        x_image = cv2.imread(x_line, 1) 
        y_image = cv2.imread(y_line, 1) 
        y_image = cv2.cvtColor(y_image, cv2.COLOR_BGR2RGB)
        x_image  = self.crop_x16(x_image)
        y_image  = self.crop_x16(y_image)
        x_image = x_image/127.5-1. #归一化x域的图片
        y_image = self.format_label(y_image)
        return x_image, y_image 

    def colorize(self, label):
        label_3c = np.stack([label]*3, axis=0)
        label_color = np.zeros_like(label_3c)
        for idx in range(1, len(self.colormap)):
            mask = (label == idx)
            for ic in range(3):
                label_color[ic][mask] = self.colormap[idx][ic]
        label_color = label_color.transpose([1,2,0])
        label_color = cv2.cvtColor(label_color.astype('uint8'), cv2.COLOR_BGR2RGB)
        return label_color


if __name__ == '__main__':
    gpu_text = subp.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    gpu_idx = np.argmax([int(x.split()[2]) for x in gpu_text.splitlines()])
    print('using......:', gpu_idx)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    run()
