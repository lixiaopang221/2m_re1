import torch.nn as nn
import torch


class M3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_filters=30, region_base=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_filters = base_filters

        self.lrelu = nn.LeakyReLU()
        # # self.dropout3d = nn.Dropout3d(p=0.6)
        self.pooling1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pooling2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pooling3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pooling4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_last = nn.Conv3d(self.base_filters, self.n_classes, kernel_size=1)
        if region_base:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        #---Level 1
        self.conv_norm_lrelu_11 = self.conv_norm_lrelu(self.in_channels ,   self.base_filters)
        self.conv_norm_lrelu_12 = self.conv_norm_lrelu(self.base_filters,   self.base_filters)
        #---Level 1 up
        self.conv_norm_lrelu_13 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters)
        self.conv_norm_lrelu_14 = self.conv_norm_lrelu(self.base_filters,   self.base_filters)

        #---Level 2
        self.conv_norm_lrelu_21 = self.conv_norm_lrelu(self.base_filters,   self.base_filters*2)
        self.conv_norm_lrelu_22 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters*2)
        #---Level 2 up 
        self.conv_norm_lrelu_23 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*2)
        self.conv_norm_lrelu_24 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters  )
        
        #---Level 3
        self.conv_norm_lrelu_31 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters*4)
        self.conv_norm_lrelu_32 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*4)
        #---Level 3 up
        self.conv_norm_lrelu_33 = self.conv_norm_lrelu(self.base_filters*8, self.base_filters*4)
        self.conv_norm_lrelu_34 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*2)
        
        #---Level 4
        self.conv_norm_lrelu_41 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*8)
        self.conv_norm_lrelu_42 = self.conv_norm_lrelu(self.base_filters*8, self.base_filters*8)
        #---Level up 4
        self.conv_norm_lrelu_43 = self.conv_norm_lrelu(self.base_filters*16, self.base_filters*8)
        self.conv_norm_lrelu_44 = self.conv_norm_lrelu(self.base_filters*8 , self.base_filters*4)
        
        #---Level 5
        self.conv_norm_lrelu_51 = self.conv_norm_lrelu(self.base_filters*8 , self.base_filters*16)
        self.conv_norm_lrelu_52 = self.conv_norm_lrelu(self.base_filters*16, self.base_filters*8 )

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU())
        return model

    def forward(self, x):
        #---Level 1
        x1 = self.conv_norm_lrelu_11(x )
        x1 = self.conv_norm_lrelu_12(x1)
        
        #---Level 2
        x2 = self.pooling1(x1)
        x2 = self.conv_norm_lrelu_21(x2)
        x2 = self.conv_norm_lrelu_22(x2)
        
        #---Level 3
        x3 = self.pooling2(x2)
        x3 = self.conv_norm_lrelu_31(x3)
        x3 = self.conv_norm_lrelu_32(x3)
        
        #---Level 4
        x4 = self.pooling3(x3)
        x4 = self.conv_norm_lrelu_41(x4)
        x4 = self.conv_norm_lrelu_42(x4)
        
        #---Level 5
        x5 = self.pooling4(x4)
        x5 = self.conv_norm_lrelu_51(x5)
        x5 = self.conv_norm_lrelu_52(x5)
        
        #---Level up 4
        x4 = torch.cat([self.upsacle(x5),x4], dim=1)
        x4 = self.conv_norm_lrelu_43(x4)
        x4 = self.conv_norm_lrelu_44(x4)
        
        #---Level up 3
        x3 = torch.cat([self.upsacle(x4),x3], dim=1)
        x3 = self.conv_norm_lrelu_33(x3)
        x3 = self.conv_norm_lrelu_34(x3)
        
        #---Level up 2
        x2 = torch.cat([self.upsacle(x3),x2], dim=1)
        x2 = self.conv_norm_lrelu_23(x2)
        x2 = self.conv_norm_lrelu_24(x2)
        
        #---Level up 1
        x1 = torch.cat([self.upsacle(x2),x1], dim=1)
        x1 = self.conv_norm_lrelu_13(x1)
        x1 = self.conv_norm_lrelu_14(x1)
        
        x0 = self.conv_last(x1)
        xs  = self.activation(x0)
        return xs, x0, x1, x2

class M2(nn.Module):
    def __init__(self, in_channels, n_classes, base_filters, region_base=False):
        super().__init__()
        self.net_0 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.net_1 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.conv_nl_1 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_2 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_f = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_last = nn.Conv3d(base_filters, n_classes, kernel_size=1)
        if region_base:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU())
        return model

    def forward(self, x):
        'x: [x0, x1, x2]'
        x0, x1 = x
        x0s, _, x01, x02 = self.net_0(x0)
        x1s, _, x11, x12 = self.net_1(x1)
        xx1 = torch.cat([x01, x11], dim=1)
        xx1 = self.conv_nl_1(xx1)
        xx2 = torch.cat([x02, x12], dim=1)
        xx2 = self.conv_nl_2(xx2)
        xx2 = self.upsacle(xx2)

        xx  = torch.cat([xx1, xx2], dim=1)
        xx  = self.conv_nl_f(xx)
        xx  = self.conv_last(xx)
        xs  = self.activation(xx)
        return x0s, x1s, xs

class M24(nn.Module):
    def __init__(self, in_channels, n_classes, base_filters, region_base=False):
        super().__init__()
        self.net_0 = M3DUNet(in_channels, n_classes-1, base_filters, region_base=region_base)
        self.net_1 = M3DUNet(in_channels, n_classes-2, base_filters, region_base=region_base)
        self.conv_nl_1 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_2 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_f = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_last = nn.Conv3d(base_filters, n_classes, kernel_size=1)
        if region_base:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU())
        return model

    def forward(self, x):
        x0s, _, x01, x02 = self.net_0(x)
        x1s, _, x11, x12 = self.net_1(x)
        xx1 = torch.cat([x01, x11], dim=1)
        xx1 = self.conv_nl_1(xx1)
        xx2 = torch.cat([x02, x12], dim=1)
        xx2 = self.conv_nl_2(xx2)
        xx2 = self.upsacle(xx2)

        xx  = torch.cat([xx1, xx2], dim=1)
        xx  = self.conv_nl_f(xx)
        xx  = self.conv_last(xx)
        xs  = self.activation(xx)
        return x0s, x1s, xs

class M2G(nn.Module):
    def __init__(self, in_channels, n_classes, base_filters, region_base=False):
        super().__init__()

        self.n_classes = n_classes
        self.net_0 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.net_1 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.conv_nl_1 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_2 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_f = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_last = nn.Conv3d(base_filters*4, n_classes, kernel_size=1)
        if region_base:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU())
        return model

    def forward(self, x):
        'x: [x0, x1, x2]'
        x0, x1 = x
        x0s, _, x01, x02 = self.net_0(x0)
        x1s, _, x11, x12 = self.net_1(x1)
        xx1 = torch.cat([x01, x11], dim=1)
        xx1 = self.conv_nl_1(xx1)
        xx2 = torch.cat([x02, x12], dim=1)
        xx2 = self.conv_nl_2(xx2)
        xx2 = self.upsacle(xx2)

        xx  = torch.cat([xx1, xx2], dim=1)
        xx  = self.conv_nl_f(xx)

        xxs = x0s + x1s
        xx_ls = []
        for i in range(self.n_classes):
            xm = xxs[:,i:i+1,...]
            xx_ls.append(xx * xm)
        xx = torch.cat(xx_ls, dim=1)

        xx  = self.conv_last(xx)
        xs  = self.activation(xx)
        return x0s, x1s, xs

class M2GA(nn.Module):
    def __init__(self, in_channels, n_classes, base_filters, region_base=False):
        super().__init__()

        self.n_classes = n_classes
        self.net_0 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.net_1 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.conv_nl_1 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_2 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_f = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_mg = nn.Conv3d(base_filters*4, base_filters, kernel_size=1)
        self.conv_last = nn.Conv3d(base_filters, n_classes, kernel_size=1)
        if region_base:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU())
        return model

    def forward(self, x):
        'x: [x0, x1, x2]'
        x0, x1 = x
        x0s, _, x01, x02 = self.net_0(x0)
        x1s, _, x11, x12 = self.net_1(x1)
        xx1 = torch.cat([x01, x11], dim=1)
        xx1 = self.conv_nl_1(xx1)
        xx2 = torch.cat([x02, x12], dim=1)
        xx2 = self.conv_nl_2(xx2)
        xx2 = self.upsacle(xx2)

        xx  = torch.cat([xx1, xx2], dim=1)
        xx  = self.conv_nl_f(xx)

        xxs = x0s + x1s
        xx_ls = []
        for i in range(self.n_classes):
            xm = xxs[:,i:i+1,...]
            xx_ls.append(xx * xm)
        xx_mg = torch.cat(xx_ls, dim=1)

        xx_mg = self.conv_mg(xx_mg)
        xx  = self.conv_last(xx_mg + xx)
        xs  = self.activation(xx)
        return x0s, x1s, xs

class M2Att(nn.Module):
    def __init__(self, in_channels, n_classes, base_filters, region_base=False):
        super().__init__()

        self.n_classes = n_classes
        self.net_0 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.net_1 = M3DUNet(in_channels, n_classes, base_filters, region_base=region_base)
        self.conv_nl_1 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_2 = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_nl_f = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.att_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        self.att_acti = nn.Sigmoid()
        self.conv_last = nn.Conv3d(base_filters, n_classes, kernel_size=1)
        if region_base:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU())
        return model

    def forward(self, x):
        'x: [x0, x1, x2]'
        x0, x1 = x
        x0s, _, x01, x02 = self.net_0(x0)
        x1s, _, x11, x12 = self.net_1(x1)
        xx1 = torch.cat([x01, x11], dim=1)
        xx1 = self.conv_nl_1(xx1)
        xx2 = torch.cat([x02, x12], dim=1)
        xx2 = self.conv_nl_2(xx2)
        xx2 = self.upsacle(xx2)

        xx  = torch.cat([xx1, xx2], dim=1)
        xx  = self.conv_nl_f(xx)

        xx_att_mean = torch.mean(xx, dim=1, keepdim=True)
        xx_att_max = torch.max(xx, dim=1, keepdim=True)[0]
        xx_att = torch.cat([xx_att_mean, xx_att_max], 1)
        xx_att = self.att_conv(xx_att)
        xx_att = self.att_acti(xx_att)
        xx = xx * xx_att

        xx  = self.conv_last(xx)
        xs  = self.activation(xx)
        return x0s, x1s, xs

class M2d(nn.Module):
    def __init__(self, in_channels, n_classes, 
                    base_filters=30, last_acti='softmax'):
        'last_acti: sigmoid, softmax, tanh'
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_filters = base_filters

        self.lrelu = nn.LeakyReLU()
        # # self.dropout3d = nn.Dropout3d(p=0.6)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_last = nn.Conv2d(self.base_filters, self.n_classes, kernel_size=1)
        if last_acti.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif last_acti.lower() == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif last_acti.lower() == 'tanh':
            self.activation = nn.Tanh()

        #---Level 1
        self.conv_norm_lrelu_11 = self.conv_norm_lrelu(self.in_channels ,   self.base_filters)
        self.conv_norm_lrelu_12 = self.conv_norm_lrelu(self.base_filters,   self.base_filters)
        #---Level 1 up
        self.conv_norm_lrelu_13 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters)
        self.conv_norm_lrelu_14 = self.conv_norm_lrelu(self.base_filters,   self.base_filters)

        #---Level 2
        self.conv_norm_lrelu_21 = self.conv_norm_lrelu(self.base_filters,   self.base_filters*2)
        self.conv_norm_lrelu_22 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters*2)
        #---Level 2 up 
        self.conv_norm_lrelu_23 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*2)
        self.conv_norm_lrelu_24 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters  )
        
        #---Level 3
        self.conv_norm_lrelu_31 = self.conv_norm_lrelu(self.base_filters*2, self.base_filters*4)
        self.conv_norm_lrelu_32 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*4)
        #---Level 3 up
        self.conv_norm_lrelu_33 = self.conv_norm_lrelu(self.base_filters*8, self.base_filters*4)
        self.conv_norm_lrelu_34 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*2)
        
        #---Level 4
        self.conv_norm_lrelu_41 = self.conv_norm_lrelu(self.base_filters*4, self.base_filters*8)
        self.conv_norm_lrelu_42 = self.conv_norm_lrelu(self.base_filters*8, self.base_filters*8)
        #---Level up 4
        self.conv_norm_lrelu_43 = self.conv_norm_lrelu(self.base_filters*16, self.base_filters*8)
        self.conv_norm_lrelu_44 = self.conv_norm_lrelu(self.base_filters*8 , self.base_filters*4)
        
        #---Level 5
        self.conv_norm_lrelu_51 = self.conv_norm_lrelu(self.base_filters*8 , self.base_filters*16)
        self.conv_norm_lrelu_52 = self.conv_norm_lrelu(self.base_filters*16, self.base_filters*8 )

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm2d(feat_out), nn.LeakyReLU(0.2))
        return model

    def forward(self, x):
        #---Level 1
        x1 = self.conv_norm_lrelu_11(x )
        x1 = self.conv_norm_lrelu_12(x1)
        
        #---Level 2
        x2 = self.pooling1(x1)
        x2 = self.conv_norm_lrelu_21(x2)
        x2 = self.conv_norm_lrelu_22(x2)
        
        #---Level 3
        x3 = self.pooling2(x2)
        x3 = self.conv_norm_lrelu_31(x3)
        x3 = self.conv_norm_lrelu_32(x3)
        
        #---Level 4
        x4 = self.pooling3(x3)
        x4 = self.conv_norm_lrelu_41(x4)
        x4 = self.conv_norm_lrelu_42(x4)
        
        #---Level 5
        x5 = self.pooling4(x4)
        x5 = self.conv_norm_lrelu_51(x5)
        x5 = self.conv_norm_lrelu_52(x5)
        
        #---Level up 4
        x4 = torch.cat([self.upsacle(x5),x4], dim=1)
        x4 = self.conv_norm_lrelu_43(x4)
        x4 = self.conv_norm_lrelu_44(x4)
        
        #---Level up 3
        x3 = torch.cat([self.upsacle(x4),x3], dim=1)
        x3 = self.conv_norm_lrelu_33(x3)
        x3 = self.conv_norm_lrelu_34(x3)
        
        #---Level up 2
        x2 = torch.cat([self.upsacle(x3),x2], dim=1)
        x2 = self.conv_norm_lrelu_23(x2)
        x2 = self.conv_norm_lrelu_24(x2)
        
        #---Level up 1
        x1 = torch.cat([self.upsacle(x2),x1], dim=1)
        x1 = self.conv_norm_lrelu_13(x1)
        x1 = self.conv_norm_lrelu_14(x1)
        
        x0 = self.conv_last(x1)
        xs = self.activation(x0)
        return xs, x0, x1, x2

class M2d3(nn.Module):
    def __init__(self, in_channels, n_classes, base_filters, last_acti='softmax'):
        super().__init__()
        self.net_0 = M2d(in_channels, n_classes, base_filters, last_acti=last_acti)
        self.net_1 = M2d(in_channels, n_classes, base_filters, last_acti=last_acti)
        self.net_2 = M2d(in_channels, n_classes, base_filters, last_acti=last_acti)
        self.conv_nl_1 = self.conv_norm_lrelu(base_filters*3, base_filters)
        self.conv_nl_2 = self.conv_norm_lrelu(base_filters*3, base_filters)
        self.conv_nl_f = self.conv_norm_lrelu(base_filters*2, base_filters)
        self.conv_last = nn.Conv2d(base_filters, n_classes, kernel_size=1)
        if last_acti.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif last_acti.lower() == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif last_acti.lower() == 'tanh':
            self.activation = nn.Tanh()

    def upsacle(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm2d(feat_out), nn.LeakyReLU())
        return model

    def forward(self, x):
        'x: [x0, x1, x2]'
        x0, x1, x2 = x
        x0s, _, x01, x02 = self.net_0(x0)
        x1s, _, x11, x12 = self.net_1(x1)
        x2s, _, x21, x22 = self.net_2(x2)
        xx1 = torch.cat([x01, x11, x21], dim=1)
        xx1 = self.conv_nl_1(xx1)
        xx2 = torch.cat([x02, x12, x22], dim=1)
        xx2 = self.conv_nl_2(xx2)
        xx2 = self.upsacle(xx2)
        xx  = torch.cat([xx1, xx2], dim=1)
        xx  = self.conv_nl_f(xx)
        xx  = self.conv_last(xx)
        xs  = self.activation(xx)
        return x0s, x1s, x2s, xs