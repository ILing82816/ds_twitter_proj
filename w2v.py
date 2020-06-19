# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:22:36 2020

@author: USER
"""
# Training word embedding
import os
import loaddata as lo
from gensim.models import word2vec

path_prefix = "D:/USA 2020 summer/Machine Learning/5 Recurrent neuron network"
model_prefix = "D:/USA 2020 summer/Machine Learning/ds_twitter_proj"

def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=0) #sg=0 CBOW, sg=1 Skip
    return model

if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = lo.load_training_data(os.path.join(path_prefix, 'training_label.txt'))

    print("loading testing data ...")
    test_x = lo.load_testing_data(os.path.join(path_prefix, 'testing_data.txt'))

    model = train_word2vec(train_x + test_x)
    
    print("saving model ...")
    model.save(os.path.join(model_prefix, 'w2v.model'))
    
    
    


