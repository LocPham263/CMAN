import tensorflow as tf
import tflearn
from keras.layers.convolutional import UpSampling3D
from tflearn.initializations import normal

from keras.layers import Activation, UpSampling3D,BatchNormalization,Add
from .spatial_transformer import Dense3DSpatialTransformer
from .utils import Network, ReLU, LeakyReLU
from tensorflow.contrib.layers import instance_norm
import keras.backend as K
import SimpleITK as sitk
import numpy as np
from .utils import Network, ReLU, LeakyReLU


def convolve(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    # print('convolve',inputLayer.get_shape(), outputChannel.get_shape())
    return tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse, weights_init=weights_init)


def convolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    # print('convolveReLU',inputLayer.get_shape(), outputChannel.get_shape())
    return ReLU(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_rectified')


def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    # print('convolveLeakyReLU',inputLayer.get_shape(), outputChannel.get_shape())
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName+'_leakilyrectified')


def upconvolve(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    # print('upconvolve',inputLayer.get_shape(), outputChannel.get_shape())
    return tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName, reuse=reuse, weights_init=weights_init)


def upconvolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False):
    # print('upconvolveRELU',inputLayer.get_shape(), outputChannel.get_shape())
    return ReLU(upconvolve(opName, inputLayer,
                           outputChannel,
                           kernelSize, stride,
                           targetShape, stddev, reuse),
                opName+'_rectified')


def upconvolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    # print('upconvolveLeakyRELU',inputLayer.get_shape(), outputChannel.get_shape())
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName+'_rectified')

def leakyReLU(inputLayer,opName, alpha = 0.1):
    return LeakyReLU(inputLayer,alpha,opName+'_leakilyrectified')

def inLeakyReLU(inputLayer,opName,alpha = 0.1):
    IN = instance_norm(inputLayer, scope=opName+'_IN')
    return LeakyReLU(IN,alpha,opName+'_leakilyrectified')

def convInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    conv = convolve(opName, inputLayer,outputChannel, kernelSize, stride, stddev, reuse)
    conv_In = instance_norm(conv, scope=opName+'_IN')
    return LeakyReLU(conv_In,alpha,opName+'_leakilyrectified')

def upconvolveInLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    up_in = instance_norm(upconvolve(opName, inputLayer,outputChannel,kernelSize, stride,targetShape, stddev, reuse), scope=opName+'_IN')
    return LeakyReLU(up_in,alpha, opName+'_rectified')

def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    attention_feature = channel_attention(input_feature, 'ch_at', ratio)
    attention_feature = spatial_attention(attention_feature, 'sp_at')
        
    return attention_feature

def channel_attention(input_feature, name, ratio=8):
    # kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    # bias_initializer = tf.constant_initializer(value=0.0)
    input_shape = input_feature.get_shape()
    channel = input_shape[-1]

    avg_pool = tf.reduce_mean(input_feature, axis=[1,2,3], keepdims=True)  
    max_pool = tf.reduce_max(input_feature, axis=[1,2,3], keepdims=True)    

    # print('avg_shape ',avg_pool.shape)
    # print('max_shape ', max_pool.shape) 
    # print('input_shape: ', input_shape)
    avg_pool = tflearn.layers.core.fully_connected(incoming=avg_pool,
                                    n_units=channel//ratio,
                                    activation='relu',
                                    # kernel_initializer=kernel_initializer,
                                    # bias_initializer=bias_initializer,
                                    name='mlp_0',
                                    reuse=False)   
    # print('avg_shape ',avg_pool.shape)
    avg_pool = tflearn.layers.core.fully_connected(incoming=avg_pool,
                                    n_units=channel,                             
                                    # kernel_initializer=kernel_initializer,
                                    # bias_initializer=bias_initializer,
                                    name='mlp_1',
                                    reuse=False)    
    # print('avg_shape ',avg_pool.shape)

    max_pool = tflearn.layers.core.fully_connected(incoming=max_pool,
                                    n_units=channel//ratio,
                                    activation='relu',
                                    # scope='mlp_0',
                                    name='mlp_2',
                                    reuse=False)   
    max_pool = tflearn.layers.core.fully_connected(incoming=max_pool,
                                    n_units=channel,
                                    # scope='mlp_1',
                                    name='mlp_3',
                                    reuse=False)  
    # print('max_shape ', max_pool.shape) 

    scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    # print(scale.get_shape())
    # print(input_feature.get_shape())
    scale = tf.expand_dims(tf.expand_dims(tf.expand_dims(scale,1),1),1)
    # print(scale.get_shape())
    return input_feature * scale
  

def spatial_attention(input_feature, name):
    kernel_size = 7
    # kernel_initializer = tf.contrib.layers.variance_scaling_initializer()

    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool,max_pool], 4)

    concat = tflearn.layers.conv_3d(concat, 1, kernel_size, 1,
                            padding="same",
                            activation='sigmoid',
                            # kernel_initializer=kernel_initializer,
                            bias=False)

    concat = tf.sigmoid(concat, 'sigmoid')
    # print(concat.get_shape())
    # print(input_feature.get_shape())
    return input_feature * concat

# This class is for CMAN without spatial attention
class CMAN(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.reconstruction = Dense3DSpatialTransformer()
    
    def build(self, img1, img2):
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        dims = 3
        c = self.channels
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      c*2,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      c*4,   3, 2)
        conv3_1 = convolveLeakyReLU('conv3_1', conv3,      c*4,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    c*8,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU('conv4_1', conv4,      c*8,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    c*16,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU('conv5_1', conv5,      c*16,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    c*32,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU('conv6_1', conv6,      c*32,  3, 1)
        # 16 * 32 = 512 channels

        shape0 = concatImgs.shape.as_list()
        shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        shape6 = conv6.shape.as_list()

        # Decoder 6
        pred6 = convolve('pred6', conv6_1, dims, 3, 1)
        upsamp6to5 = upconvolve('upsamp6to5', pred6, dims, 4, 2, shape5[1:4])

        deconv5 = upconvolveLeakyReLU(
            'deconv5', conv6_1, shape5[4], 4, 2, shape5[1:4])
        deconv5 = channel_attention(deconv5,'deconv5_att')
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        
        # Decoder 5
        pred5 = convolve('pred5', concat5, dims, 3, 1)
        upsamp5to4 = self.reconstruction([pred5, upsamp6to5])
        upsamp5to4 = upconvolve('upsamp5to4', upsamp5to4, dims, 4, 2, shape4[1:4])

        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        deconv4 = channel_attention(deconv4,'deconv4_att')
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2


        # Decoder 4
        pred4 = convolve('pred4', concat4, dims, 3, 1)
        upsamp4to3 = self.reconstruction([pred4, upsamp5to4])
        upsamp4to3 = upconvolve('upsamp4to3', upsamp4to3, dims, 4, 2, shape3[1:4])
        
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        deconv3 = channel_attention(deconv3,'deconv3_att')
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2


        # Decoder 3
        pred3 = convolve('pred3', concat3, dims, 3, 1)
        upsamp3to2 = self.reconstruction([pred3, upsamp4to3])
        upsamp3to2 = upconvolve('upsamp3to2', upsamp3to2, dims, 4, 2, shape2[1:4])
        
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        deconv2 = channel_attention(deconv2,'deconv2_att')
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2


        # Decoder 2
        pred2 = convolve('pred2', concat2, dims, 3, 1)
        upsamp2to1 = self.reconstruction([pred2, upsamp3to2])
        upsamp2to1 = upconvolve('upsamp2to1', upsamp2to1, dims, 4, 2, shape1[1:4])
        
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        deconv1 = channel_attention(deconv1,'deconv1_att')
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')

        pred1 = convolve('pred1', concat1, dims, 3, 1)
        upsamp1to0 = self.reconstruction([pred1, upsamp2to1])
        pred0 = upconvolve('upsamp1to0', upsamp1to0, dims, 4, 2, shape0[1:4])

        return {'flow': pred0 * 20 * self.flow_multiplier}
        # return {'flow': pred0}

# This class is for CMAN without channel attention
class CMAN_SA(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.reconstruction = Dense3DSpatialTransformer()
    
    def build(self, img1, img2):
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        dims = 3
        c = self.channels
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      c*2,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      c*4,   3, 2)
        conv3_1 = convolveLeakyReLU('conv3_1', conv3,      c*4,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    c*8,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU('conv4_1', conv4,      c*8,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    c*16,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU('conv5_1', conv5,      c*16,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    c*32,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU('conv6_1', conv6,      c*32,  3, 1)
        # 16 * 32 = 512 channels

        shape0 = concatImgs.shape.as_list()
        shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        shape6 = conv6.shape.as_list()

        # Decoder 6
        pred6 = convolve('pred6', conv6_1, dims, 3, 1)
        pred6_att = spatial_attention(pred6, 'pred6_att')
        upsamp6to5 = upconvolve('upsamp6to5', pred6_att, dims, 4, 2, shape5[1:4])

        deconv5 = upconvolveLeakyReLU(
            'deconv5', conv6_1, shape5[4], 4, 2, shape5[1:4])
        deconv5 = spatial_attention(deconv5,'deconv5_att')
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        
        # Decoder 5
        pred5 = convolve('pred5', concat5, dims, 3, 1)
        pred5_att = spatial_attention(pred5, 'pred5_att')
        upsamp5to4 = self.reconstruction([pred5_att, upsamp6to5])
        upsamp5to4 = upconvolve('upsamp5to4', upsamp5to4, dims, 4, 2, shape4[1:4])

        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        deconv4 = spatial_attention(deconv4,'deconv4_att')
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2


        # Decoder 4
        pred4 = convolve('pred4', concat4, dims, 3, 1)
        pred4_att = spatial_attention(pred4, 'pred4_att')
        upsamp4to3 = self.reconstruction([pred4_att, upsamp5to4])
        upsamp4to3 = upconvolve('upsamp4to3', upsamp4to3, dims, 4, 2, shape3[1:4])
        
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        deconv3 = spatial_attention(deconv3,'deconv3_att')
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2


        # Decoder 3
        pred3 = convolve('pred3', concat3, dims, 3, 1)
        pred3_att = spatial_attention(pred3, 'pred3_att')
        upsamp3to2 = self.reconstruction([pred3_att, upsamp4to3])
        upsamp3to2 = upconvolve('upsamp3to2', upsamp3to2, dims, 4, 2, shape2[1:4])
        
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        deconv2 = spatial_attention(deconv2,'deconv2_att')
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2


        # Decoder 2
        pred2 = convolve('pred2', concat2, dims, 3, 1)
        pred2_att = spatial_attention(pred2, 'pred2_att')
        upsamp2to1 = self.reconstruction([pred2_att, upsamp3to2])
        upsamp2to1 = upconvolve('upsamp2to1', upsamp2to1, dims, 4, 2, shape1[1:4])
        
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        deconv1 = spatial_attention(deconv1,'deconv1_att')
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')

        pred1 = convolve('pred1', concat1, dims, 3, 1)
        pred1_att = spatial_attention(pred1, 'pred1_att')
        upsamp1to0 = self.reconstruction([pred1_att, upsamp2to1])
        pred0 = upconvolve('upsamp1to0', upsamp1to0, dims, 4, 2, shape0[1:4])

        return {'flow': pred0 * 20 * self.flow_multiplier}
        # return {'flow': pred0}

# This class is for CMAN (you can change the name of the class for your convenience)
class CMAN_CA(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.reconstruction = Dense3DSpatialTransformer()

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        dims = 3
        c = self.channels
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      c*2,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      c*4,   3, 2)
        conv3_1 = convolveLeakyReLU('conv3_1', conv3,      c*4,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    c*8,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU('conv4_1', conv4,      c*8,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    c*16,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU('conv5_1', conv5,      c*16,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    c*32,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU('conv6_1', conv6,      c*32,  3, 1)
        # 16 * 32 = 512 channels

        shape0 = concatImgs.shape.as_list()
        shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        shape6 = conv6.shape.as_list()
        print('Shape: ',shape0, shape1, shape2, shape3, shape4, shape5)


        # Decoder 6
        pred6 = convolve('pred6', conv6_1, dims, 3, 1)
        pred6_att = spatial_attention(pred6, 'pred6_att')
        upsamp6to5 = upconvolve('upsamp6to5', pred6_att, dims, 4, 2, shape5[1:4])

        deconv5 = upconvolveLeakyReLU(
            'deconv5', conv6_1, shape5[4], 4, 2, shape5[1:4])
        deconv5 = cbam_block(deconv5,'deconv5_att')
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        
        # Decoder 5
        pred5 = convolve('pred5', concat5, dims, 3, 1)
        pred5_att = spatial_attention(pred5, 'pred5_att')
        upsamp5to4 = self.reconstruction([pred5_att, upsamp6to5])
        upsamp5to4 = upconvolve('upsamp5to4', upsamp5to4, dims, 4, 2, shape4[1:4])

        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        deconv4 = cbam_block(deconv4,'deconv4_att')
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2


        # Decoder 4
        pred4 = convolve('pred4', concat4, dims, 3, 1)
        pred4_att = spatial_attention(pred4, 'pred4_att')
        upsamp4to3 = self.reconstruction([pred4_att, upsamp5to4])
        upsamp4to3 = upconvolve('upsamp4to3', upsamp4to3, dims, 4, 2, shape3[1:4])
        
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        deconv3 = cbam_block(deconv3,'deconv3_att')
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2


        # Decoder 3
        pred3 = convolve('pred3', concat3, dims, 3, 1)
        pred3_att = spatial_attention(pred3, 'pred3_att')
        upsamp3to2 = self.reconstruction([pred3_att, upsamp4to3])
        upsamp3to2 = upconvolve('upsamp3to2', upsamp3to2, dims, 4, 2, shape2[1:4])
        
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        deconv2 = cbam_block(deconv2,'deconv2_att')
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2


        # Decoder 2
        pred2 = convolve('pred2', concat2, dims, 3, 1)
        pred2_att = spatial_attention(pred2, 'pred2_att')
        upsamp2to1 = self.reconstruction([pred2_att, upsamp3to2])
        upsamp2to1 = upconvolve('upsamp2to1', upsamp2to1, dims, 4, 2, shape1[1:4])
        
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        deconv1 = cbam_block(deconv1,'deconv1_att')
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')

        pred1 = convolve('pred1', concat1, dims, 3, 1)
        pred1_att = spatial_attention(pred1, 'pred1_att')
        upsamp1to0 = self.reconstruction([pred1_att, upsamp2to1])
        pred0 = upconvolve('upsamp1to0', upsamp1to0, dims, 4, 2, shape0[1:4])

        # pred1 = convolve('pred1', concat1, dims, 3, 1)
        # upsam1to0 = upconvolve('upsamp1to0', pred1, dims, 4, 2, shape0[1:4])
        # pred = spatial_attention(upsam1to0, 'pred1_att')
        # pred = self.reconstruction([upsam1to0, upsamp2to1])

        # sc = 32
        # upsamp6to5 = tf.reshape(upsamp6to5, [1,int(128/sc),1,int(128/sc),1,int(128/sc),1,3])
        # upsamp6to5 = tf.tile(upsamp6to5, [1,1,sc,1,sc,1,sc,1])
        # upsamp6to5 = tf.reshape(upsamp6to5, [1,128,128,128,3])

        return {'flow': pred0 * 20 * self.flow_multiplier}
        # return {'flow': pred0}
        # return {'flow': upsamp6to5}
        # return {'flow': tf.tile(pred5, [1,sc,sc,sc,1]) * 20 * self.flow_multiplier}


def affine_flow(W, b, len1, len2, len3):
    b = tf.reshape(b, [-1, 1, 1, 1, 3])
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr * wx + yr * wy) + (zr * wz + b)

def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])

class VTNAffineStem(Network):
    def __init__(self, name, flow_multiplier=1., **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'coloncatImgs')

        dims = 3
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, 16,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      32,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      64,   3, 2)
        conv3_1 = convolveLeakyReLU(
            'conv3_1', conv3,      64,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    128,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU(
            'conv4_1', conv4,      128,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    256,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU(
            'conv5_1', conv5,      256,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    512,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU(
            'conv6_1', conv6,      512,  3, 1)
        ks = conv6_1.shape.as_list()[1:4]
        conv7_W = tflearn.layers.conv_3d(
            conv6_1, 9, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_W')
        conv7_b = tflearn.layers.conv_3d(
            conv6_1, 3, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_b')

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = tf.reshape(conv7_W, [-1, 3, 3]) * self.flow_multiplier
        b = tf.reshape(conv7_b, [-1, 3]) * self.flow_multiplier
        A = W + I
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x
        # the model learns W = A - I.

        sx, sy, sz = img1.shape.as_list()[1:4]
        flow = affine_flow(W, b, sx, sy, sz)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = tf.nn.l2_loss(det - 1.0)
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = tf.matmul(A, A, True) + epsI

        def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = tf.add_n([M[0][0], M[1][1], M[2][2]])
            sigma2 = tf.add_n([
                M[0][0] * M[1][1],
                M[1][1] * M[2][2],
                M[2][2] * M[0][0]
            ]) - tf.add_n([
                M[0][1] * M[1][0],
                M[1][2] * M[2][1],
                M[2][0] * M[0][2]
            ])
            sigma3 = tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])
            return sigma1, sigma2, sigma3
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = tf.reduce_sum(ortho_loss)

        return {
            'flow': flow,
            'W': W,
            'b': b,
            'det_loss': det_loss,
            'ortho_loss': ortho_loss
        }
