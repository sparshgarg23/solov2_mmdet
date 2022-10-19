# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:34:47 2022

@author: Admin
"""

import torch.nn as nn
from mmcv.cnn import xavier_init
import torch.nn.functional as F

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule

@NECKS.register_module
class FPN(nn.Module):
    
    def __init__(self,in_channel,out_channel,num_outs,
                 start_level=0,end_level=1,add_extra_conv=False,extra_conv_on_inputs=True,
                 relu_before=False,no_norm_on=False,conv_cfg=None,norm_cfg=None,activation=None):
        super(FPN,self).__init__()
        assert isinstance(in_channel,list)
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.num_ins=len(in_channel)
        self.num_out=num_outs
        self.activation=activation
        self.relu_before=relu_before
        self.no_norm_on=no_norm_on
        self.fp16_enabled=False
        
        if end_level==-1:
            self.backbone_end_level=self.num_ins
            assert num_outs>=self.num_ins-start_level
        else:
            self.backbone_end_level=end_level
            assert end_level<=len(in_channel)
            assert num_outs==end_level-start_level
        self.start_level=start_level
        self.end_level=end_level
        self.add_extra_conv=add_extra_conv
        self.extra_conv_on_inputs=extra_conv_on_inputs
        
        self.lateral_convs=nn.ModuleList()
        self.fpn_convs=nn.ModuleList()
        
        for i in range(self.start_level,self.backbone_end_level):
            l_conv=ConvModule(in_channel[i],out_channel,1,conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg if not self.no_norm_on else None,
                              activation=self.activation,
                              inplace=False)
            fpn_conv=ConvModule(out_channel,out_channel,3,padding=1,
                                conv_cfg=conv_cfg,norm_cfg=norm_cfg,
                                activation=self.activation,
                                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels=num_outs-self.backbone_end_level+self.start_level
        if add_extra_conv and extra_levels>=1:
            for  i in range(extra_levels):
                if i==0 and self.extra_conv_on_inputs:
                    in_channel=self.in_channel[self.backbone_end_level-1]
                else:
                    in_channel=out_channel
                extra_fpn_conv=ConvModule(in_channel,out_channel,3,stride=2,padding=1,
                                          conv_cfg=conv_cfg,norm_cfg=norm_cfg,activation=self.activation,
                                          inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                xavier_init(m,distribution='uniform')
    @auto_fp16()
    def forward(self,inputs):
        assert len(inputs)==len(self.in_channels)
        #build lateral
        lateral=[lateral_conv(inputs[i+self.start_level])
                 for i,lateral_conv in enumerate(self.lateral_convs)]
        #Build topdown path
        used_backbone_level=len(lateral)
        for i in range(used_backbone_level-1,0,-1):
            lateral[i-1]+=F.interpolate(
                lateral[i],scale_factor=2,mode='nearest')
        #build output
        outs=[self.fpn_convs[i](lateral[i]) for i in range(used_backbone_level)]
        
        if self.num_outs>len(outs):
            if not self.add_extra_conv:
                for i in range(self.num_outs-used_backbone_level):
                    outs.append(F.max_pool2d(outs[-1],1,stride=2))
            else:
                if self.extra_conv_on_inputs:
                    orig=inputs[self.backbone_end_level-1]
                    outs.append(self.fpn_convs[used_backbone_level](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_level](outs[-1]))
                
                for i in range(used_backbone_level+1,self.num_outs):
                    if self.relu_before:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
    
                
                
                
                    