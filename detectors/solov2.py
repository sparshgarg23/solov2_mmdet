# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:08:05 2022

@author: Admin
"""

from .single_stage_ins import SingleStageInsDetector
from ..registry import DETECTORS

@DETECTORS.register_module

class SOLOv2(SingleStageInsDetector):
    def __init__(self,backbone,neck,bbox_head,mask_feat_head,
                 train_cfg=None,test_cfg=None,pretrained=None):
        super(SOLOv2,self).__init__(backbone,neck,bbox_head,mask_feat_head,train_cfg,
                                    test_cfg,pretrained)
        