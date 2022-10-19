# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:39:14 2022

@author: Admin
"""

import torch
import torch.nn as nn

class Scale(nn.Module):
    
    def __init__(self,scale=1.0):
        super(Scale,self).__init__()
        self.scale=nn.Parameter(torch.tensor(scale,dtype=torch.float))
    def forward(self,x):
        return x*self.scale
    