from distutils.version import LooseVersion

import numpy as np
import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, weight_map=None, 
                    epsilon=1e-5, reduce_dim=[2,3,4], channel_wise=False):
        super().__init__()
        self.epsilon    = epsilon
        self.weight_map = weight_map
        self.reduce_dim = reduce_dim
        self.channel_wise = channel_wise

    def dice_channel(self, pred, target):
        if self.weight_map is not None:
            pred = pred*self.weight_map
        pred_sum = pred.sum(self.reduce_dim)
        target_sum = target.sum(self.reduce_dim)
        intersection = (pred * target).sum(self.reduce_dim)
        dice = (2 * intersection + self.epsilon) / (pred_sum + target_sum + self.epsilon)
        # loss_dice = 1 - dice.mean()
        return dice.mean()

    def dice_coe(self, pred, target):
        pred_sum = (pred*pred).sum(self.reduce_dim)
        target_sum = (target*target).sum(self.reduce_dim)
        intersection = (pred * target).sum(self.reduce_dim)
        dice = (2 * intersection + self.epsilon) / (pred_sum + target_sum + self.epsilon)
        # loss_dice = 1 - dice.mean()
        return dice.mean()

    def forward(self, pred, target):
        if self.channel_wise:
            chns = int(target.shape[1]) #---channels number
            dice_chns = []
            for chn in range(chns):
                pred_c = pred[:,chn:chn+1,...]
                target_c = target[:,chn:chn+1,...]
                dice_chns.append(self.dice_channel(pred_c,target_c))
            dice_chns = torch.stack(dice_chns)
            dice_mean = torch.mean(dice_chns)
            dice_loss = 1 - dice_mean
            return dice_loss
        else:
            dice =  self.dice_coe(pred, target)
            dice_mean = torch.mean(dice)
            dice_loss = 1 - dice_mean
            return dice_loss

class CrossEntropy(torch.nn.Module):
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')
        super().__init__()

    def forward(self, pred, label):
        'label: map with label(not binary).'
        label = label.view(-1)
        pred = pred.permute([0,2,3,4,1]).contiguous()
        pred = pred.view(-1, pred.shape[-1])
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        return criterion(pred, label)

class FocalLoss(torch.nn.Module):
    def __init__(self, class_num, alpha_ls, gama_ls, ignore_bg=False):
        'if ignore background, the x must be one more channel than y'
        super().__init__()
        assert(len(alpha_ls)==len(gama_ls)==class_num), "params is imcompatable!"
        self.class_num = class_num
        self.alpha_ls = alpha_ls
        self.gama_ls = gama_ls
        self.ignore_bg = ignore_bg

    def forward(self, x, y):
        'x: probability maps; y: onehot ground truth'
        if self.ignore_bg:
            x = x[...,1:]
        li_sum = 0
        for i in range(0, self.class_num):
            pi = (1 - x[:,i,...])**self.gama_ls[i]
            li = self.alpha_ls[i] * y[:,i,...] * pi * torch.log(x[:,i,...])
            li_sum += torch.sum(li)
        return -li_sum

class BCEWithLogitsLoss2d(torch.nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), \
                    "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), \
                    "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), \
                    "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.autograd.Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    predict, target, weight=weight, size_average=self.size_average)
        return loss

class CrossEntropy2d(torch.nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), \
                    "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), \
                    "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} \
                    ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.autograd.Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = torch.nn.functional.cross_entropy(
                    predict, target, weight=weight, size_average=self.size_average)
        return loss

class DiceNp():
    def __init__(self, num_class=4):
        self.num_class = num_class

    def __call__(slef, *ls, **kdict):
        return self.dice(*ls, **kdict)

    def dice(self, pred, gt, num_class=None, e=1e-5):
        if num_class is not None:
            self.num_class = num_class
        pred = pred.astype('int')
        pred = pred.flatten()
        pred = self.one_hot(pred)
        gt = gt.astype('int')
        gt = gt.flatten()
        gt = self.one_hot(gt)
        insec = np.sum(gt * pred, axis=0)  #---intersection
        #---当predict和gt中某类为0时，dice = 1, e为epsilon
        dice = (2.*insec + e)/(np.sum(gt, axis=0) + np.sum(pred, axis=0) + e)
        return dice

    def dice_wt_tc_et(self, pred, gt, num_class=4, e=1e-5):
        '''the label format of pred and gt: 
            wt: [1,2,3]; tc: [2,3]; et: [3]
            num_class: inclue background
        '''
        dices = []
        for i in range(num_class - 1):
            pred_i = np.zeros_like(pred)
            pred_i[pred > i] = 1
            gt_i = np.zeros_like(gt)
            gt_i[gt > i] = 1
            dices.append(self.dice_binary(pred_i, gt_i))
        return dices

    def dice_binary(self, pred, gt, e=1e-5):
        insec = np.sum(gt * pred)  #---intersection
        #---当predict和gt中某类为0时，dice = 1, e为epsilon
        dice = (2.*insec + e)/(np.sum(gt) + np.sum(pred) + e)
        return dice

    def one_hot(self, x, label_ls=None, axis=-1):
        if label_ls is None:
            label_ls = list(range(1, self.num_class))
        out = []
        for i in label_ls:
            temp = np.zeros_like(x)
            temp[x == i] = 1
            out.append(temp)
        return np.stack(out, axis=axis)

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = torch.functional.log_softmax(input)
    else:
        # >=0.3
        log_p = torch.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = torch.functional.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss
