from keras.models import Model
from keras.layers import Dense, Input, Embedding
from keras.layers import GlobalMaxPool1D, Dropout, Conv1D, BatchNormalization, Activation, Add, SeparableConv1D


def block(x, kernel_size, filters):
    x_Conv_1 = SeparableConv1D(filters=filters, kernel_size=[kernel_size], strides=1,
                        padding="same", kernel_initializer="he_uniform")(x)
    x_Conv_2 = Add()([x, x_Conv_1])
    x = Activation(activation='relu')(x_Conv_2)
    return x
