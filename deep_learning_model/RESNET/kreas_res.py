import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
from keras.layers import BatchNormalization, Concatenate
import pandas as pd
import time
from keras.models import load_model
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(r"./")
from resnet import block
from evaluate import predict2top, f1_avg, Metrics
from textcnn import textcnn_one
from keras.utils import multi_gpu_model
import gc
import keras
from attention import attention

np.random.seed(1)

print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

num_words = 400000
maxlen = 1500
DIM = 256
print('num_words = {}, maxlen = {}'.format(num_words, maxlen))

# 数据集和标签
fact = np.load(r"../data/train_word_seg_data{}_numwords{}.npy".format(maxlen, num_words))
labels = np.load(r"../data/train_label_from_zero_onehot.npy")
fact_train, fact_test, labels_train, labels_test = train_test_split(fact, labels, test_size=0.1,
                                                                    random_state=1)

print("starting training")
metrics = Metrics()
data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words + 1,
                     input_length=maxlen,
                     output_dim=DIM,
                     mask_zero=False,
                     name='Embedding',
                     )(data_input)

block1 = block(x=word_vec, kernel_size=16, filters=DIM)
block2 = block(x=word_vec, kernel_size=15, filters=DIM)

max = GlobalMaxPool1D()(block1)
avg = GlobalAveragePooling1D()(block1)

max2 = GlobalMaxPool1D()(block2)
avg2 = GlobalAveragePooling1D()(block2)

x = Concatenate(axis=1)([max, avg, max2, avg2])
x = BatchNormalization()(x)
x = Dense(1000, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(labels_train.shape[1], activation="sigmoid")(x)
model = Model(inputs=data_input, outputs=x)
model.summary()
model = multi_gpu_model(model, gpus=2)
adam = keras.optimizers.adam(lr=0.00001)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks_list = [metrics,
                  keras.callbacks.EarlyStopping(monitor='acc', patience=3, ),
                  keras.callbacks.ModelCheckpoint(filepath=r'../output/my_model_res_aug_.h5',
                                                  monitor='f1_avg_val',
                                                  save_best_only=True, ),
                  keras.callbacks.ReduceLROnPlateau(monitor='f1_avg_val', factor=0.1,
                                                    patience=2, ),
                  ]

model.fit(x=fact_train, y=labels_train, batch_size=128, epochs=100, verbose=1,
          validation_data=[fact_test, labels_test], callbacks=callbacks_list)
