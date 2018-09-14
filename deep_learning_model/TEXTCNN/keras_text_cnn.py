import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
import pandas as pd
from keras.models import load_model
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(r"./")
from evaluate import predict2top, f1_avg, Metrics
from cnn_bin import cnn
from keras.utils import multi_gpu_model
import gc
import keras

np.random.seed(1)
num_words = 400000
maxlen = 1500
filters = 512
print('num_words = {}, maxlen = {}'.format(num_words, maxlen))

# 数据集和标签
fact = np.load(r"../data/train_word_seg_data{}_numwords{}.npy".format(maxlen, num_words))
labels = np.load(r"../data/train_label_from_zero_onehot.npy")
fact_train, fact_test, labels_train, labels_test = train_test_split(fact, labels, test_size=0.1,
                                                                    random_state=1)
# 数据增强
augment_word_seg = np.load(r"../data/augment800_word_seg_data1500_numwords400000.npy")
augment_label = np.load(r"../data/augment_800_label_from_zero_onehot.npy")

print(labels_train.shape)
print(augment_label.shape)

augment_fact_train = np.vstack((fact_train, augment_word_seg))
augment_labels_train = np.vstack((labels_train, augment_label))

print(augment_fact_train.shape)
print(augment_labels_train.shape)

del augment_label
del labels
del fact
del augment_word_seg
gc.collect()
print("data have been loaded")

print("starting training")
metrics = Metrics()
data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words + 1,
                     input_length=maxlen,
                     output_dim=512,
                     mask_zero=False,
                     name='Embedding',
                     # weights=[embedding_weights],
                     # trainable=False,
                     )(data_input)
x1 = cnn(word_vec=word_vec, kernel_size=15, filters=filters)
x2 = cnn(word_vec=word_vec, kernel_size=16, filters=filters)
x3 = cnn(word_vec=word_vec, kernel_size=17, filters=filters)

conat = Concatenate(axis=1)([x1, x2, x3])
x = BatchNormalization()(x1)
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
                  keras.callbacks.ModelCheckpoint(filepath=r'../output/my_model_textcnn_aug.h5',
                                                  monitor='f1_avg_val',
                                                  save_best_only=True, ),
                  keras.callbacks.ReduceLROnPlateau(monitor='f1_avg_val', factor=0.1,
                                                    patience=2, ), ]

model.fit(x=fact_train, y=labels_train, batch_size=256, epochs=100, verbose=1,
          validation_data=[fact_test, labels_test], callbacks=callbacks_list)
