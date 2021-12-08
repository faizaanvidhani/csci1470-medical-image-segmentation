import os
#import numpy as numpy
import time
import datetime
import tensorflow as tf
import inspect

class up_conv(tf.keras.Model):
    def __init__(self,ch_in,ch_out):
        super(up_conv, self).__init__()
        self.up = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(size=(2,2)),
            tf.keras.layers.Conv2D(filters=ch_out,kernel_size=3,strides=(2,2),padding='same',use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

    def call(self,x):
        x = self.up(x)
        return x


class Recurrent_block(tf.keras.Model):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()

        self.t = t
        self.ch_out = ch_out
        #self.filter = tf.Variable(tf.random.truncated_normal([3,3,ch_out,ch_out],stddev=0.1))
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=ch_out,kernel_size=3,strides = (1,1),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])
        

    def call(self,x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
                #print("x1 shape before", x1.shape)
            #print("x shape", x.shape)
            x1 = self.conv(x+x1)
            #print("x1 shape after:", x1.shape)
        return x1

class RRCNN_block(tf.keras.Model):
    def __init__(self,ch_in, ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = tf.keras.Sequential([
            Recurrent_block(ch_out,t),
            Recurrent_block(ch_out,t)
        ])
        self.Conv_1x1 = tf.keras.layers.Conv2D(filters=ch_out,kernel_size=2,strides=(1,1), padding='same')

    def call(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class R2U_Net(tf.keras.Model):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()

        self.Maxpool = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides=(1,1), padding= 'same')
        self.Upsample = tf.keras.layers.UpSampling2D(size=(2,2))

        """ self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t) """

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=16,t=t)
        self.RRCNN2 = RRCNN_block(ch_in=16,ch_out=32,t=t)
        self.RRCNN3 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        self.RRCNN4 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        self.RRCNN5 = RRCNN_block(ch_in=128,ch_out=256,t=t)

        """ self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up2 = up_conv(ch_in=128,ch_out=64) """

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Up4 = up_conv(ch_in=128,ch_out=64)
        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Up2 = up_conv(ch_in=32,ch_out=16)

        """ self.Up_RRCNN5 = RRCNN_block(ch_in=1024,ch_out=512,t=t)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512,ch_out=256,t=t)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256,ch_out=128,t=t)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128,ch_out=64,t=t) """

        self.Up_RRCNN5 = RRCNN_block(ch_in=256,ch_out=128,t=t)
        self.Up_RRCNN4 = RRCNN_block(ch_in=128,ch_out=64,t=t)
        self.Up_RRCNN3 = RRCNN_block(ch_in=64,ch_out=32,t=t)
        self.Up_RRCNN2 = RRCNN_block(ch_in=32,ch_out=16,t=t)

        #self.Conv_1x1 = tf.keras.layers.Conv2D(64,kernel_size=1,strides=(1,1),padding='same')
        self.Conv_1x1 = tf.keras.layers.Conv2D(3,kernel_size=1,strides=(1,1),padding='same')


    def call(self,x):
        #encoding path
        x1 = self.RRCNN1(x)
        print("x1 shape", x1.shape)

        x2 = self.Maxpool(x1)
        print("x2 shape after maxpool", x2.shape)
        x2 = self.RRCNN2(x2)
        print("x2 shape", x2.shape)

        x3 = self.Maxpool(x2)
        print("x3 shape after maxpool", x3.shape)
        x3 = self.RRCNN3(x3)
        print("x3 shape", x3.shape)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        print("x4 shape", x4.shape)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        print("x5 shape", x5.shape)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = tf.concat((x4,d5),axis=0)
        d5 = self.Up_RRCNN5(d5)
        print("d5.shape", d5.shape)

        d4 = self.Up4(d5)
        d4 = tf.concat((x3,d4),axis=0)
        d4 = self.Up_RRCNN4(d4)
        print("d4 shape", d4.shape)

        d3 = self.Up3(d4)
        d3 = tf.concat((x2,d3),axis=0)
        d3 = self.Up_RRCNN3(d3)
        print("d3 shape", d3.shape)

        d2 = self.Up2(d3)
        d2 = tf.concat((x1,d2),axis=0)
        d2 = self.Up_RRCNN2(d2)
        print("d2 shape", d2.shape)

        d1 = self.Conv_1x1(d2)
        print("d1 shape", d1.shape)

        return d1