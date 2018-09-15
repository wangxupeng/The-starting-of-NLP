import gc
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
import pandas as pd
import time
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(r"./")
from keras.models import load_model
from evaluate import predict2top, f1_avg, Metrics
import keras
from keras.utils import multi_gpu_model

print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
print('accusation')

print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

num_words = 400000
maxlen = 1500
kernel_size = 3
DIM = 256
filters = 256
print('num_words = {}, maxlen = {}'.format(num_words, maxlen))

# 数据集和标签
fact = np.load(r"../data/train_word_seg_data{}_numwords{}.npy".format(maxlen, num_words))
labels = np.load(r"../data/train_label_from_zero_onehot.npy")
fact_train, fact_test, labels_train, labels_test = train_test_split(fact, labels, test_size=0.1,
                                                                    random_state=1)
del labels
del fact
gc.collect()
print("data have been loaded")

metrics = Metrics()
data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words + 1,
                     input_length=maxlen,
                     output_dim=DIM,
                     mask_zero=0,
                     name='Embedding')(data_input)
x = Bidirectional(CuDNNGRU(filters, return_sequences=True))(word_vec)
x = Bidirectional(CuDNNGRU(filters, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
x = BatchNormalization()(x)
x = Dense(labels_train.shape[1], activation="sigmoid")(x)
model = Model(inputs=data_input, outputs=x)
model.summary()
model = multi_gpu_model(model, gpus=2)
adam = keras.optimizers.adam(lr=0.0001)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=3, ),
                  keras.callbacks.ModelCheckpoint(filepath=r'../output/my_model_bigru_fasttext.h5',
                                                  monitor='val_acc',
                                                  save_best_only=True, ),
                  keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                                    patience=2, ),
                  metrics,
                  ]

model.fit(x=fact_train, y=labels_train, batch_size=128, epochs=100, verbose=1,
          validation_data=[fact_test, labels_test], callbacks=callbacks_list)
