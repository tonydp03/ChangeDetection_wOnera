# coding: utf-8

"""
Change Detection models for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Fri Dec 13, 2019
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
    y_true = y_true[:, :, :, -1]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return ((2. * intersection + smooth) / (union + smooth))  # not working better using mean


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def weighted_bce_dice_loss(y_true,y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])

    class_weights = [0.1, 0.9]
    weighted_bce = K.sum(class_loglosses * K.constant(class_weights))

    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work
    return weighted_bce + 0.5 * (dice_coef_loss(y_true, y_pred))


def UNetPP_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):   
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x0 = x
    x = BatchNormalization(name='bn' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_2', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn' + stage + '_2')(x)
    if mode == 'residual':
        x = Add(name='resi' + stage)([x, x0])
    return x


def EF_UNetPP(input_shape, classes=1):
    mode='residual'
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    inputs = Input(shape=input_shape, name='input')
    
    conv1_1 = UNetPP_ConvUnit(inputs, stage='11', nb_filter=nb_filter[0], mode=mode)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1) 
    
    conv2_1 = UNetPP_ConvUnit(pool1, stage='21', nb_filter=nb_filter[1], mode=mode)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = UNetPP_ConvUnit(conv1_2, stage='12', nb_filter=nb_filter[0], mode=mode)

    conv3_1 = UNetPP_ConvUnit(pool2, stage='31', nb_filter=nb_filter[2], mode=mode)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = UNetPP_ConvUnit(conv2_2, stage='22', nb_filter=nb_filter[1], mode=mode)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = UNetPP_ConvUnit(conv1_3, stage='13', nb_filter=nb_filter[0], mode=mode)

    conv4_1 = UNetPP_ConvUnit(pool3, stage='41', nb_filter=nb_filter[3], mode=mode)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = UNetPP_ConvUnit(conv3_2, stage='32', nb_filter=nb_filter[2], mode=mode)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = UNetPP_ConvUnit(conv2_3, stage='23', nb_filter=nb_filter[1], mode=mode)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = UNetPP_ConvUnit(conv1_4, stage='14', nb_filter=nb_filter[0], mode=mode)

    conv5_1 = UNetPP_ConvUnit(pool4, stage='51', nb_filter=nb_filter[4], mode=mode)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = UNetPP_ConvUnit(conv4_2, stage='42', nb_filter=nb_filter[3], mode=mode)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = UNetPP_ConvUnit(conv3_3, stage='33', nb_filter=nb_filter[2], mode=mode)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = UNetPP_ConvUnit(conv2_4, stage='24', nb_filter=nb_filter[1], mode=mode)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = UNetPP_ConvUnit(conv1_5, stage='15', nb_filter=nb_filter[0], mode=mode)

#    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1',
#                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
#    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2',
#                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
#    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3',
#                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
#    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4',
#                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    conv_fuse = concatenate([conv1_2, conv1_3, conv1_4, conv1_5], name='merge_fuse', axis=bn_axis)
    output = Conv2D(classes, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv_fuse)
    
    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(lr=1e-4), loss=weighted_bce_dice_loss, metrics=['accuracy'])
    
    return model


def UNet_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):   
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
#    x0 = x
#    x = BatchNormalization(name='bn' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_2', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn' + stage)(x)
#    x = BatchNormalization(name='bn' + stage + '_2')(x)
    if mode == 'residual':
        x = Add(name='resi' + stage)([x, x0])
    return x


def EF_UNet(input_shape, classes=1):
    mode = 'None'
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    inputs = Input(shape=input_shape, name='input')

    conv1 = UNet_ConvUnit(inputs, stage='1', nb_filter=nb_filter[0], mode=mode)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UNet_ConvUnit(pool1, stage='2', nb_filter=nb_filter[1], mode=mode)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UNet_ConvUnit(pool2, stage='3', nb_filter=nb_filter[2], mode=mode)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = UNet_ConvUnit(pool3, stage='4', nb_filter=nb_filter[3], mode=mode)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom of the U-Net
    conv5 = UNet_ConvUnit(pool4, stage='5', nb_filter=nb_filter[4], mode=mode)
    
    # Right side of the U-Net
    up1 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(conv5)
    merge1 = concatenate([conv4,up1], axis=bn_axis)
    conv6 = UNet_ConvUnit(merge1, stage='6', nb_filter=nb_filter[3], mode=mode)

    up2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(conv6)
    merge2 = concatenate([conv3,up2], axis=bn_axis)
    conv7 = UNet_ConvUnit(merge2, stage='7', nb_filter=nb_filter[2], mode=mode)

    up3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(conv7)
    merge3 = concatenate([conv2,up3], axis=bn_axis)
    conv8 = UNet_ConvUnit(merge3, stage='8', nb_filter=nb_filter[1], mode=mode)
    
    up4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(conv8)
    merge4 = concatenate([conv1,up4], axis=bn_axis)
    conv9 = UNet_ConvUnit(merge4, stage='9', nb_filter=nb_filter[0], mode=mode)

    # Output layer of the U-Net with a softmax activation
    output = Conv2D(classes, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv9)

    model = Model(input=inputs, output=output)
    model.compile(optimizer=Adam(lr=1e-4), loss ='binary_crossentropy', metrics = ['accuracy'])    
    # model.compile(optimizer=Adam(lr=1e-4), loss =weighted_bce_dice_loss, metrics = ['accuracy'])    

    return model
