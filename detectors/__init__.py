# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:10:34 2022

@author: Admin
"""

from .base import BaseDetector
from .single_stage_ins import SingleStageInsDetector
from .solov2 import SOLOv2

__all__=['BaseDetector','SingleStageInsDetector','SOLOv2']
