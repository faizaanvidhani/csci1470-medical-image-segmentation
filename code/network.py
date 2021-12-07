import tensorflow as tf
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.nn import init 

'''

def init_weights(net, init_type='normal', gain=0.02):

    
    the init function may not be compeletely necessary
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
'''




class conv_block(tf.keras.Model):
    '''
    Purpose: The block reduces the image dimensionality as it would in typical convolutional layers. This is important for identifying the notable parts of the segmented item; i.e. finding the 'what'
    Input: Instantiated with number of in channels and out channels
    Returns: If 'forward' method is called, it will run convolution 

    #CONCERNS
    batch_normalization from pytorch.BatchNorm2d=> the 'feature' input and clarifying which of the tf paraemeters (mean,variance, epsilon, variance epsilon) need to be explicitly defined
    '''
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = tf.keras.Sequential(
            tf.nn.conv2d(filters = [3,3,ch_in,ch_out], strides=1,padding=1),#(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True) is the original paraemeters; switched up to make it tf friendly 
            tf.batch_normalization(),#tf.nn.BatchNorm2d(ch_out),
            tf.nn.relu(),#tf.nn.ReLU(inplace=True)
            tf.nn.conv2d(filters = [3,3,ch_in,ch_out], strides=1,padding=1),
            tf.batch_normalization(),#tf.nn.BatchNorm2d(ch_out),
            tf.nn.relu()
        )



def forward(self,x):
	x = self.conv(x)
	return x


class up_conv(tf.keras.Model):
    '''
    Purpose: The block increases dimensionality; i.e. the 'building/reconstructive portion' of the u-net architecture. This is important for identifying the 'where' of the segemented lesion
    Input: Instantiated with number of in channels and out channels
    Returns: If 'forward' method is called, it will run upsampling  

    #CONCERNS
    upsampling from pytorch.upsample => the scale factor is not a parameter in the tf equivalent, so we will need to determine the output dimensions manually

    '''
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = tf.keras.Sequential(
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.nn.conv2d(filters = [3,3,ch_in,ch_out], strides=1,padding=1),#tf.nn.conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            tf.batch_normalization(),#tf.nn.BatchNorm2d(ch_out),
            tf.nn.relu(),#tf.nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(tf.keras.Model):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = tf.keras.Sequential(
            tf.nn.conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            tf.batch_normalization(ch_out),
            tf.nn.relu()
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(tf.keras.Model):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = tf.keras.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = tf.nn.conv2d(filters=[1,1,ch_in,ch_out],strides=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(tf.keras.Model):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = tf.keras.Sequential(
            tf.nn.conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            tf.nn.BatchNorm2d(ch_out),
            tf.nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class R2U_Net(tf.keras.Model):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = tf.nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = tf.keras.layers.UpSampling2D(size=(2, 2)),

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = tf.nn.conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


def forward(self,x):
	# encoding path
	x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = tf.concat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = tf.concat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = tf.concat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = tf.concat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

