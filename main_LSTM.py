# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:08:51 2020

@author: USER
"""
import os
import torch
import loaddata as lo
import preprocess as pre
import model_LSTM as m
import data as d
import train as tr
from torch.utils.data import DataLoader
import predict as p
import pandas as pd

# Set up some path and parameters.
## path
data_prefix = 'D:/USA 2020 summer/Machine Learning/5 Recurrent neuron network'
train_with_label = os.path.join(data_prefix, 'training_label.txt')
testing_data = os.path.join(data_prefix, 'testing_data.txt')

path_prefix = "D:/USA 2020 summer/Machine Learning/ds_twitter_proj"
w2v_path = os.path.join(path_prefix, 'w2v.model')

model_dir = path_prefix # model directory for checkpoint model 

## defined the length of sentence、fit_embedding、the size of batch、epoch、learning rate
sen_len = 20
fix_embedding = True # fix embedding during training
batch_size = 128
epoch =5
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.is_available(), if it return the "True", then device will be "cuda", if it return the "False", then device will be "cpu".





#load data
print("loading data ...")
train_x, y = lo.load_training_data(train_with_label)





#data preprocess
## use w2v.model get embedding
preprocess = pre.Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

## train_test split
X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

##To dataset for Dataloader
train_dataset = d.TwitterDataset(X=X_train, y=y_train)
val_dataset = d.TwitterDataset(X=X_val, y=y_val)

## To batch of tensors
train_loader = DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

val_loader = DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)





# Set up model and train model
print("Start training...")
model = m.LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=5, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device) 
tr.training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)





#Prediction
print("loading testing data ...")
test_x = lo.load_testing_data(testing_data)
preprocess = pre.Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = d.TwitterDataset(X=test_x, y=None)
test_loader = DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'twitter_LSTM.model'))
outputs = p.testing(batch_size, test_loader, model, device)

## output save to cvs
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(os.path.join(path_prefix, 'LSTM_predict.csv'), index=False)



print("Finish Predicting")
