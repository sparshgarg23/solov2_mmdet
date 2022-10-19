# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:56:23 2022

@author: Admin
"""

import warnings
import torch.nn as nn
from mmcv.cnn import constant_init,kaiming_init

from mmdet.ops import DeformConvPack,ModulatedDeformConvPack
from .conv_ws import ConvWS2d
from .norm import build_norm_layer

conv_cfg={
    'Conv':nn.Conv2d,
    'ConvWS':ConvWS2d,
    'DCN':DeformConvPack,
    'DCNv2':ModulatedDeformConvPack
    }

def build_conv_layer(cfg,*args,**kwargs):
    """
    Build convolutional layer
    args: cfg (None or dict)
    Return nn.layer
    """
    
    if cfg is None:
        cfg_=dict(type='Conv')
    else:
        assert isinstance(cfg,dict) and 'type' in cfg
        cfg_=cfg.copy()
    layer_type=cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer=conv_cfg[layer_type]
    layer=conv_layer(*args,**kwargs,**cfg_)
    return layer

class ConvModule(nn.Module):
    """
    Conv block
    in_channel,out_channel,kernel_size,stride,
    padding,dilation,groups,bias
    conv_cfg,norm_cfg dict for specifing config for conv and norm
    activation str ReLU by defualt
    inplace wether to use inplace mode
    order the order of conv/norm/activation type tuple of strings
    """
    
    def __init__(self,in_channel,out_channel,kernel_size,
                 stride=1,padding=0,dilation=1,groups=1,
                 bias='auto',conv_cfg=None,norm_cfg=None,activation='relu',inplace=True,
                 order=('conv','norm','act')):
        super(ConvModule,self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg,dict)
        assert norm_cfg is None or isinstance(norm_cfg,dict)
        
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.activation=activation
        self.inplace=inplace
        self.order=order
        
        assert isinstance(self.order,tuple) and len(self.order)==3
        assert set(order)==set(['conv','norm','act'])
        
        self.with_norm=norm_cfg is not None
        self.with_activation=activation is not None
        
        if bias=='auto':
            bias=False if self.with_norm else True
        self.with_bias=True
        
        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')
        
        self.conv=build_conv_layer(conv_cfg, in_channel, out_channel,kernel_size,
                                   stride=stride,padding=padding,dilation=dilation,
                                   groups=groups,bias=bias)
        
        self.in_channels=self.conv.in_channels
        self.out_channels=self.conv.out_channels
        self.kernel_size=self.conv.kernel_size
        self.stride=self.conv.stride
        self.padding=self.conv.padding
        self.dilation=self.conv.dilation
        self.transposed=self.conv.transposed
        self.output_padding=self.conv.output_padding
        self.groups=self.conv.groups
        
        if self.with_norm:
            if order.index('norm') >order.index('conv'):
                norm_channels=out_channel
            else:
                norm_channels=in_channel
        self.norm_name,norm=build_norm_layer(norm_cfg,norm_channels)
        self.add_module(self.norm_name,norm)
        
        if self.with_activation:
            if self.activation not in ['relu']:
                raise ValueError('{} is curently not supported.'.format(self.activation))
            if self.activation=='relu':
                self.activate=nn.ReLU(inplace=inplace)
        
        self.init_weights()
    @property
    def norm(self):
        return getattr(self,self.norm_layer)
    def init_weights(self):
        nonlinearity='relu' if self.activation is None else self.activation
        kaiming_init(self.conv,nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm,1,bias=0)
    def forward(self,x,activate=True,norm=True):
        for layer in self.order:
            if layer=='conv':
                x=self.conv(x)
            elif layer=='norm' and norm and self.with_norm:
                x=self.norm(x)
            elif layer=='act' and activate and self.with_activation:
                x=self.activate(x)
            return x
        
        
        