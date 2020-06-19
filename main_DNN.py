# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:47:01 2020

@author: USER
"""
import os
import torch
import loaddata as lo
import BOW as BO
import model_DNN as m
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

model_dir = path_prefix # model directory for checkpoint model 

## defined the length of sentence、the size of batch、epoch、learning rate
max_len = 1200
batch_size = 128
epoch =5
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.is_available(), if it return the "True", then device will be "cuda", if it return the "False", then device will be "cpu".





#load data
print("loading data ...")
train_x, y = lo.load_training_data(train_with_label)
test_x = lo.load_testing_data(testing_data)




#data preprocess
## use bag of word that don't consider the order
max_len = 1200
b = BO.BOW(max_len=max_len)
b.bow(train_x, test_x)
train_x = b['train']
test_x = b['test']

y = [int(label) for label in y]
y = torch.LongTensor(y)

## train_test split
X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

##To dataset for Dataloader
train_dataset = d.TwitterDataset(X=X_train, y=y_train)
val_dataset = d.TwitterDataset(X=X_val, y=y_val)
test_dataset = d.TwitterDataset(X=test_x, y=None)

## To batch of tensors
train_loader = DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
val_loader = DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)
test_loader = DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)





# Set up model and train model
print("Start training...")
model = m.DNN_Net(embedding_dim=max_len, num_layers=1)
model = model.to(device) 
tr.training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)





#Prediction
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'twitter_DNN.model'))
outputs = p.testing(batch_size, test_loader, model, device)
## output save to cvs
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(os.path.join(path_prefix, 'DNN_predict.csv'), index=False)



print("Finish Predicting")

