import os
import time
import keras
import keras.backend as K
import numpy as np
from models import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import Embedding, Dense

# hyper-parameters
batch_size = 32
learning_rate = 0.02
validation_split_ratio = 0.1
model_save_directory = "./"

# load data
train_X = []
train_Y = []
test_X = []

with open("data/train_10.csv") as file:
    for line in file:
        X = [int(x) for x in line.strip().split(",")]
        train_X.append(X)
            
with open("data/train_8.csv") as file:
    for line in file:
        X = [int(x) for x in line.strip().split(",")]
        if len(X) < 8:
            train_Y.append([0] * (8 - len(X)) + X)
        else:
            train_Y.append(X)

with open("data/test_10.csv") as file:
    for line in file:
        X = [int(x) for x in line.strip().split(",")]
        test_X.append(X)          

cut = int(validation_split_ratio * len(train_X))

train_X, val_X = train_X[:-cut], train_X[-cut:]
train_Y, val_Y = train_Y[:-cut], train_Y[-cut:]

train_inp = np.empty((len(train_X), 7, 10))
train_tar = np.empty((len(train_Y), 8, 8))
val_inp = np.empty((len(val_X), 7, 10))
val_tar = np.empty((len(val_Y), 8, 8))
test_inp = np.empty((len(test_X), 7, 10))

for i,x in enumerate(train_X):
    train_inp[i,] = to_categorical(x, 10)

for i,x in enumerate(train_Y):
    train_tar[i,] = to_categorical(x, 8)

for i,x in enumerate(val_X):
    val_inp[i,] = to_categorical(x, 10)

for i,x in enumerate(val_Y):
    val_tar[i,] = to_categorical(x, 8)

for i,x in enumerate(test_X):
    test_inp[i,] = to_categorical(x, 10)
    
# model
model = Sequential()

seq2seq = AttentionSeq2Seq(input_dim=10, input_length=7, hidden_dim=256, output_length=8, output_dim=128, depth=1)

model.add(seq2seq)

model.add(Dense(8, activation="softmax"))

adam = keras.optimizers.Adam(lr = learning_rate)

model.compile(loss="categorical_crossentropy", optimizer=adam)

# print(model.summary())

# train
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

STAMP = 'seq2seq_%d' % (batch_size)

checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

bst_model_path = checkpoint_dir + STAMP + '.h5'

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

model.fit(train_inp, train_tar, 
          validation_data=(val_inp, val_tar), 
          epochs=30, batch_size=32, shuffle=True, 
          callbacks=[early_stopping, model_checkpoint, tensorboard])

# predict forward
ans = []

preds_forward = model.predict(test_inp, verbose=1)

# model.reset_states()

# reload
train_inp_inv = np.empty((len(train_X), 7, 10))
train_tar_inv = np.empty((len(train_Y), 8, 8))
val_inp_inv = np.empty((len(val_X), 7, 10))
val_tar_inv = np.empty((len(val_Y), 8, 8))
test_inp_inv = np.empty((len(test_X), 7, 10))

for i,x in enumerate(train_X):
    train_inp_inv[i,] = to_categorical(x[::-1], 10)

for i,x in enumerate(train_Y):
    train_tar_inv[i,] = to_categorical(x[::-1], 8)

for i,x in enumerate(val_X):
    val_inp_inv[i,] = to_categorical(x[::-1], 10)

for i,x in enumerate(val_Y):
    val_tar_inv[i,] = to_categorical(x[::-1], 8)

for i,x in enumerate(test_X):
    test_inp_inv[i,] = to_categorical(x[::-1], 10)
    
# re-train
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

STAMP = 'seq2seq_%d' % (batch_size)

checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

bst_model_path = checkpoint_dir + STAMP + '.h5'

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

model.fit(train_inp_inv, train_tar_inv, 
          validation_data=(val_inp_inv, val_tar_inv), 
          epochs=30, batch_size=32, shuffle=True, 
          callbacks=[early_stopping, model_checkpoint, tensorboard])

# predict backward
pred_backward = model.predict(test_inp_inv, verbose=1)

preds = preds_forward + np.flip(pred_backward, axis=1)

for p in preds:
    temp = [str(np.argmax(vec)) for vec in p]
    index = -1
    for i,x in enumerate(temp):
        if x != "0":
            index = i
            break
    ans.append(temp[i:])

with open("data/test_8.csv", "w") as file:
    for line in ans:
        file.write(",".join(line)+"\n")
