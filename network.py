import torch
import torch.nn as nn

# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvLayer, self).__init__()
        paddings = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(paddings)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups) #, padding) 
        # self.in_d = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ConvLayer_dpws(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer_dpws, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels)
        self.in_1d = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        self.in_2d = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.in_1d(self.conv1(x))
        out = self.relu(self.in_2d(self.conv2(out)))
        return out


class ConvLayer_dpws_last(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer_dpws_last, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels)
        self.in_1d = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        # self.in_2d = nn.InstanceNorm2d(out_channels, affine=True)
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.in_1d(self.conv1(x))
        # out = self.relu(self.in_2d(self.conv2(out)))
        out = self.conv2(out)
        return out

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.in_d = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer_dpws(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer_dpws, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size, stride, groups=in_channels)
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True )
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True )
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        # out = self.reflection_pad(x)
        # out = self.conv2d(out)
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out


class DeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeConvLayer, self).__init__()

        # reflection_padding = kernel_size // 2
        # self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1)
    def forward(self, x):
        # out = self.reflection_pad(x)
        out = self.deconv2d(x)
        return out

# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
       
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)


    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        # out = self.relu(self.in2(self.conv2(out)))
        out = self.in2(self.conv2(out))
        # out = self.relu(self.conv2(out))
        # out = self.conv2(out)
        out = out + residual
        # out = self.relu(out)
       
        return out 


class ResidualBlock_depthwise(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_depthwise, self).__init__()
       
        # ########################## deptwise ###########################################
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=channels)
        self.in1 = nn.InstanceNorm2d(channels, affine=True )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True )

        self.conv3 = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=channels)
        self.in3 = nn.InstanceNorm2d(channels, affine=True )
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.in4 = nn.InstanceNorm2d(channels, affine=True )

        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()

    def forward(self, x):
        
        # ############### DEPTWISE ###################
        # residual = x
        # out = self.relu(self.in1(self.conv1(x)))
        # out = self.relu(self.in2(self.conv2(out)))
        # out = self.relu(self.in3(self.conv3(out)))
        # out = self.relu(self.in4(self.conv4(out)))
        # out = out + residual
         
        # # ################## v1 ####################
        # residual = x
        # out = self.in1(self.conv1(x))
        # out = self.relu(self.in2(self.conv2(out)))
        # out = self.in3(self.conv3(out))
        # out = self.in4(self.conv4(out))
        # out = out + residual
        # out = self.relu(out) 

        # ################## v2 ####################  √
        residual = x
        out = self.in1(self.conv1(x))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.in3(self.conv3(out))
        out = self.in4(self.conv4(out))
        out = out + residual 

        # ################## v3 ####################
        # residual = x
        # out = self.conv1(x)
        # out = self.relu(self.in2(self.conv2(out)))
        # out = self.conv3(out)
        # out = self.in4(self.conv4(out))
        # out = out + residual

        # ################## v4 ####################
        # residual = x
        # out = self.in1(self.conv1(x))
        # out = self.relu(self.in2(self.conv2(out)))
        # out = self.in3(self.conv3(out))
        # out = self.relu(self.in4(self.conv4(out)))
        # out = out + residual
        
        return out 


# Image Transform Network
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True )

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True )

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True )

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # self.res6 = ResidualBlock(128)

        # decoding layers
        # TODO:
        # self.deconv3 = DeConvLayer(128, 64, kernel_size=3, stride=2)
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in3_d = nn.InstanceNorm2d(64, affine=True )

        # self.deconv2 = DeConvLayer(64, 32, kernel_size=3, stride=2)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in2_d = nn.InstanceNorm2d(32, affine=True )

        self.deconv1 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True )

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))
        # y = self.relu(self.conv1(x))
        # y = self.relu(self.conv2(y))
        # y = self.relu(self.conv3(y))
        y_downsample = y

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        
        y_upsample = y
        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        # y = self.relu(self.deconv3(y))
        # y = self.relu(self.deconv2(y))
        # y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        # return y, y_downsample, y_upsample
        return y


ALAPHA_1 = 0.25
ALAPHA_2 = 0.25
# ALAPHA_1 = 0.5
# ALAPHA_2 = 0.5
class ImageTransformNet_dpws(nn.Module):
    def __init__(self):
        super(ImageTransformNet_dpws, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer_dpws(3, int(32*ALAPHA_1), kernel_size=9, stride=1)
        # self.in1_e = nn.InstanceNorm2d(int(32*ALAPHA_1), affine=True )
        
        self.conv2 = ConvLayer_dpws(int(32*ALAPHA_1), int(64*ALAPHA_1), kernel_size=3, stride= 2)
        self.conv3 = ConvLayer_dpws(int(64*ALAPHA_1), int(128*ALAPHA_2), kernel_size=3, stride= 2)

        # residual layers
        self.res1 = ResidualBlock_depthwise(int(128*ALAPHA_2))
        self.res2 = ResidualBlock_depthwise(int(128*ALAPHA_2))
        self.res3 = ResidualBlock_depthwise(int(128*ALAPHA_2))
        self.res4 = ResidualBlock_depthwise(int(128*ALAPHA_2))
        self.res5 = ResidualBlock_depthwise(int(128*ALAPHA_2))
        # self.res6 = ResidualBlock_depthwise(128)

        # decoding layers
        # TODO:
        # self.deconv3 = DeConvLayer(128, 64, kernel_size=3, stride=2)
        self.deconv3 = UpsampleConvLayer_dpws(int(128*ALAPHA_2), int(64*ALAPHA_1), kernel_size=3, stride=1, upsample=2)
        # self.in3_d = nn.InstanceNorm2d(int(64*ALAPHA_1), affine=True )

        # self.deconv2 = DeConvLayer(64, 32, kernel_size=3, stride=2)
        self.deconv2 = UpsampleConvLayer_dpws(int(64*ALAPHA_1), int(32*ALAPHA_1), kernel_size=3, stride=1, upsample=2)
        # self.in2_d = nn.InstanceNorm2d(32, affine=True )

        self.deconv1 = ConvLayer_dpws_last(int(32*ALAPHA_1), 3, kernel_size=9, stride=1)
        # self.deconv1 = ConvLayer_dpws_last(int(32*ALAPHA_1), 3, kernel_size=9, stride=1)
        # self.in1_d = nn.InstanceNorm2d(3, affine=True )

    def forward(self, x):
        # encode
        # y = self.relu(self.in1_e(self.conv1(x)))
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y_downsample = y

        # y = self.relu(self.in2_e(self.conv2(y)))
        # y = self.relu(self.in3_e(self.conv3(y)))
        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        # y = self.res6(y)
        y_upsample = y

        # decode
        y = self.deconv3(y)
        y = self.deconv2(y)
        y = self.deconv1(y)

        # return y, y_downsample, y_upsample
        return y

class distiller_1(nn.Module):
    def __init__(self):
        super(distiller_1, self).__init__()
        
        self.conv = nn.Conv2d(128, int(128*ALAPHA_2), kernel_size=1, stride=1)

    def forward(self, x):
        # encode
        y = self.conv(x)

        return y

class distiller_2(nn.Module):
    def __init__(self):
        super(distiller_2, self).__init__()
        
        self.conv = nn.Conv2d(128, int(128*ALAPHA_2), kernel_size=1, stride=1)

    def forward(self, x):
        # encode
        y = self.conv(x)

        return y
