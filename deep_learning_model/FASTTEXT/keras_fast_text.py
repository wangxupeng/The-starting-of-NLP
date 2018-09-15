import gc
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
import pandas as pd
import time
import sys
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(r"./")
from keras.models import load_model
from evaluate import predict2top, f1_avg, Metrics
from attention import attention
import keras
from keras.utils import multi_gpu_model
from keras.initializers import *
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

num_words = 400000
maxlen = 1500
DIM = 56
print('num_words = {}, maxlen = {}'.format(num_words, maxlen))

# 数据集和标签
fact = np.load(r"../data/train_word_seg_data{}_numwords{}.npy".format(maxlen, num_words))
labels = np.load(r"../data/train_label_from_zero_onehot.npy")
fact_train, fact_test, labels_train, labels_test = train_test_split(fact, labels, test_size=0.1,
                                                                    random_state=1)

print("data have been loaded")

metrics = Metrics()
data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words + 1,
                     input_length=maxlen,
                     output_dim=DIM,
                     mask_zero=0,
                     name='fast_text_Embedding',
                     embeddings_initializer=tf.contrib.layers.xavier_initializer(), )(data_input)

x1 = GlobalAveragePooling1D(name="fast_text_GAP")(word_vec)
x2 = GlobalMaxPooling1D(name="fast_text_GMP")(word_vec)

x = Concatenate(axis=1)([x1, x2])
x = BatchNormalization(name="text_cnn_batch_norm")(x)

x = Dense(labels_train.shape[1],
          activation="sigmoid",
          name="text_cnn_ouput",
          kernel_initializer=tf.contrib.layers.xavier_initializer())(x)

model = Model(inputs=data_input, outputs=x)
model.summary()
model = multi_gpu_model(model, gpus=2)
adam = keras.optimizers.adam(lr=0.00001)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks_list = [metrics,
                  keras.callbacks.EarlyStopping(monitor='acc', patience=3, ),
                  keras.callbacks.ModelCheckpoint(filepath=r'../output/my_model_fasttext_aug.h5',
                                                  monitor='f1_avg_val',
                                                  save_best_only=True, ),
                  keras.callbacks.ReduceLROnPlateau(monitor='f1_avg_val', factor=0.1,
                                                    patience=2, ), ]
model.fit(x=fact_train, y=labels_train, batch_size=128, epochs=100, verbose=1,
          validation_data=[fact_test, labels_test], callbacks=callbacks_list)

