# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:43:10 2020

@author: USER
"""
import torch
from torch import nn
import torch.optim as optim
import loaddata as lo

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    t_batch = len(train) 
    v_batch = len(valid)
    
    loss = nn.BCELoss() #loss function: binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr) # optimizer: adam
    best_acc = 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        
        # training
        model.train() 
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) 
            labels = labels.to(device, dtype=torch.float) 
            
            optimizer.zero_grad() 
            outputs = model(inputs) 
            outputs = outputs.squeeze() 
            batch_loss = loss(outputs, labels) 
            batch_loss.backward() 
            optimizer.step() 
            
            correct = lo.evaluation(outputs, labels) 
            total_acc += (correct / batch_size)
            total_loss += batch_loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, batch_loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # validation
        model.eval() 
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long) 
                labels = labels.to(device, dtype=torch.float) 
                outputs = model(inputs) 
                outputs = outputs.squeeze() 
                batch_loss = loss(outputs, labels) 
                
                correct = lo.evaluation(outputs, labels) 
                total_acc += (correct / batch_size)
                total_loss += batch_loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "{}/twitter_LSTM.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
