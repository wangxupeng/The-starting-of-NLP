from keras.layers import *
import sys

def cnn_one(word_vec=None, kernel_size=1, filters=512):
    x = word_vec
    x = SeparableConv1D(filters=filters, kernel_size=[kernel_size], strides=1,
                        padding="same", kernel_initializer="he_uniform")(x)
    x = Activation(activation='relu')(x)
    x = GlobalMaxPool1D()(x)
    return x


