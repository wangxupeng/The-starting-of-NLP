from keras.layers import *
from keras.models import *

def attention(input=None, depth=None):
    attention = Dense(1, activation='tanh')(input)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(depth)(attention)
    attention = Permute([2, 1], name='attention_vec')(attention)
    attention_mul = Multiply(name='attention_mul')([input, attention])
    return attention_mul
