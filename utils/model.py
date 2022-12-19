import torch.nn as nn
import torch


class M3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, 
                    base_filters=30, last_acti='Sigmoid'):
        'last_acti: sigmoid, softmax, tanh'
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
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x

    def conv_norm_lrelu(self, feat_in, feat_out, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU(0.2))
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
        
        x1 = self.conv_last(x1)
        x  = self.activation(x1)
        return x, x1

class Disc(nn.Module):
    def __init__(self, in_channels, n_classes=1, base_filters=32):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.lrelu = nn.LeakyReLU()
        self.conv_1 = nn.Conv3d(
                    self.in_channels, self.base_filters, kernel_size=4, stride=2, padding=1)
        self.conv_n_l_1 = self.conv_norm_lrelu(self.base_filters, 2*self.base_filters, stride=2)
        self.conv_n_l_2 = self.conv_norm_lrelu(self.base_filters*2, 4*self.base_filters, stride=2)
        self.conv_n_l_3 = self.conv_norm_lrelu(self.base_filters*4, 8*self.base_filters, stride=2)
        self.conv_2 = nn.Conv3d(self.base_filters*8, self.n_classes, kernel_size=4, padding=1)

    def conv_norm_lrelu(self, feat_in, feat_out, stride=1, padding=1, bias=True):
        model = nn.Sequential(
                    nn.Conv3d(feat_in, feat_out, kernel_size=4, 
                            stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm3d(feat_out), nn.LeakyReLU(0.2, True))
        return model

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_n_l_1(x)
        x = self.conv_n_l_2(x)
        x = self.conv_n_l_3(x)
        x = self.conv_2(x)
        return x

class Disc2d(nn.Module):
    def __init__(self, in_channels, n_classes=1, base_filters=32):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_filters = base_filters
        self.lrelu = nn.LeakyReLU()
        self.conv_1 = nn.Conv2d(
                    self.in_channels, self.base_filters, kernel_size=4, stride=2, padding=1)
        self.conv_n_l_1 = self.conv_norm_lrelu(self.base_filters, 2*self.base_filters, stride=2)
        self.conv_n_l_2 = self.conv_norm_lrelu(self.base_filters*2, 4*self.base_filters, stride=2)
        self.conv_n_l_3 = self.conv_norm_lrelu(self.base_filters*4, 8*self.base_filters, stride=2)
        self.conv_2 = nn.Conv2d(self.base_filters*8, self.n_classes, kernel_size=4, padding=1)

    def conv_norm_lrelu(self, feat_in, feat_out, stride=1, padding=1, bias=True):
        model = nn.Sequential(
                    nn.Conv2d(feat_in, feat_out, kernel_size=4, 
                            stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm2d(feat_out), nn.LeakyReLU(0.2, True))
        return model

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_n_l_1(x)
        x = self.conv_n_l_2(x)
        x = self.conv_n_l_3(x)
        x = self.conv_2(x)
        return x

class M2d(nn.Module):
    def __init__(self, in_channels, n_classes, 
                    base_filters=30, last_acti='Sigmoid'):
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
