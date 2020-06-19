# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:02:27 2020

@author: USER
"""
import torch
from torch import nn

class DNN_Net(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super(DNN_Net, self).__init__()
        self.num_layers = num_layers
        self.classifier = nn.Sequential( nn.Linear(embedding_dim, 512),
                                         nn.Linear(512, 128),
                                         nn.Linear(128, 1),
                                         nn.Sigmoid())
    def forward(self, inputs):
        x = self.classifier(inputs.float())
        return x


