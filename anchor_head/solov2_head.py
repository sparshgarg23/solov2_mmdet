# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:46:43 2022

@author: Admin
"""

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import DeformConv,roi_align
from mmdet.core import multi_apply,matrix_nms

from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob,ConvModule

INF=1e8

def center_of_mass(bitmasks):
    _,h,w=bitmasks.size()
    ys=torch.arange(0,h,dtype=torch.float32,device=bitmasks.device)
    xs=torch.arange(0,w,dtype=torch.float32,device=bitmasks.device)
    
    m00=bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10=(bitmasks*xs).sum(dim=-1).sum(dim=-1)
    m01=(bitmasks*ys[:,None]).sum(dim=-1).sum(dim=-1)
    center_x=m10/m00
    center_y=m01/m00
    return center_x,center_y

def points_nms(heat,kernel=2):
    hmax=nn.functional.max_pool2d(heat,(kernel,kernel),stride=1,padding=1)
    keep=(hmax[:,:,:-1,:-1]==heat).float()
    return heat*keep

def dice_loss(inputs,target):
    inputs=inputs.contiguous().view(inputs.size()[0], -1)
    target=target.contiguous().view(target.size()[0],-1).float()
    
    a=torch.sum(inputs*target,1)
    b=torch.sum(inputs*inputs,1)+0.001
    c=torch.sum(target*target,1)+0.001
    d=(2*a)/(b+c)
    return 1-d

@HEADS.register_module
class SOLOv2Head(nn.Module):
    
    def __init__(self,num_classes,in_channels,seg_feat_channels=256,
                 stacked_conv=4,strides=(4,8,16,32,64),
                 base_edge_list=(16,32,64,128,256),
                 scale_ranges=((8,32),(16,64),(32,128),(64,256),(128,512)),
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=64,
                 loss_ins=None,loss_cate=None,conv_cfg=None,
                 norm_cfg=None,use_dcn_in_tower=None,type_dcn=None):
        super(SOLOv2Head,self).__init__()
        self.num_classes=num_classes
        self.seg_num_grids=num_grids
        self.cate_out_channel=self.num_classes-1
        self.ins_out_channel=ins_out_channels
        self.in_channels=in_channels
        
        self.seg_feat_channels=seg_feat_channels
        self.stacked_convs=stacked_conv
        self.strides=strides
        self.sigma=sigma
        self.stacked_convs=stacked_conv
        
        self.kernel_out_channel=self.ins_out_channel*1*1
        self.base_edge_list=base_edge_list
        self.scale_range=scale_ranges
        self.loss_cate=build_loss(loss_cate)
        self.ins_loss_weight=loss_ins['loss_weight']
        
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.use_dcn_in_tower=use_dcn_in_tower
        self.type_dcn=type_dcn
        
        self._init_layers()
    
    def _init_layers(self):
        norm_cfg=dict(type='GN',num_groups=32,requires_grad=True)
        
        self.cate_convs=nn.ModuleList()
        self.kernel_convs=nn.ModuleList()
        
        for i in range(self.stacked_convs):
            
            if self.use_dcn_in_tower:
                cfg_conv=dict(type=self.type_dcn)
            else:
                cfg_conv=self.conv_cfg
            
            chn=self.in_channels+2 if i==0 else self.seg_feat_channels
            self.kernel_convs.append(
                ConvModule(chn,self.seg_feat_channels,
                           3,stride=1,padding=1,
                           conv_cfg=cfg_conv,norm_cfg=norm_cfg,
                           bias=norm_cfg is None
                    )
                )
            chn=self.in_channels if i==0 else self.seg_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,self.seg_feat_channels,
                    3,stride=1,padding=1,
                    conv_cfg=cfg_conv,norm_cfg=norm_cfg,
                    bias=norm_cfg is None)
                )
            self.solo_cate=nn.Conv2d(
                self.seg_feat_channels,self.cate_out_channel,
                3,padding=1
                )
            self.solo_kernel=nn.Conv2d(self.seg_feat_channels,
                                       self.kernel_out_channel,
                                       3,padding=1)
        def init_weights(self):
            for m in self.cate_convs:
                normal_init(m.conv,std=0.01)
            for  m in self.kernel_convs:
                normal_init(m.conv,std=0.01)
            bias_cate=bias_init_with_prob(0.01)
            normal_init(self.solo_cate,std=0.01,bias=bias_cate)
            normal_init(self.solo_kernel,std=0.01)
        
        def forward(self,feats,eval=False):
            new_feats=self.split_feats(feats)
            featmap_size=[featmap.size()[-2:] for featmap in new_feats]
            upsampled_size=(featmap_size[0][0]*2,featmap_size[0][1]*2)
            cate_pred,kernel_pred=multi_apply(self.forward_single,new_feats,
                                              list(range(len(self.seg_num_grids
                                               ))),
                                              eval=eval,upsampled_size=upsampled_size)
            return cate_pred,kernel_pred
        
        def split_feats(self,feats):
            return (F.interpolate(feats[0],scale_factor=0.5,mode='bilinear'),
                    feats[1],feats[2],feats[3],
                    F.interpolate(feats[4],size=feats[3].shape[-2:],mode='bilinear'))
        def forward_single(self,x,idx,eval=False,upsampled_size=None):
            ins_kernel_feat=x
            x_range=torch.linspace(-1,1,ins_kernel_feat.shape[-1],device=ins_kernel_feat.device)
            y_range=torch.linsapce(-1,1,ins_kernel_feat.shape[-2],device=ins_kernel_feat.device)
            y,x=torch.meshgrid(y_range,x_range)
            y=y.expand([ins_kernel_feat.shape[0],1,-1,-1])
            x=x.expand([ins_kernel_feat.shape[0],1,-1,-1])
            coord_feat=torch.cat([x,y],1)
            ins_kernel_feat=torch.cat([ins_kernel_feat,coord_feat],1)
            
            #KERNEL BRANCH
            kernel_feat=ins_kernel_feat
            seg_num_grid=self.seg_num_grids[idx]
            kernel_feat=F.interpolate(kernel_feat,size=seg_num_grid,mode='bilinear')
            
            cate_feat=kernel_feat[:,:-2,:,:]
            kernel_feat=kernel_feat.contiguous()
            
            for i,kernel_layer in enumerate(self.kernel_convs):
                kernel_feat=kernel_layer(kernel_feat)
            kernel_pred=self.solo_kernel(kernel_feat)
            
            #CATE BRANCH
            cate_feat=cate_feat.contiguous()
            for i,cate_layer in enumerate(self.cate_convs):
                cate_feat=cate_layer(cate_feat)
            cate_pred=self.solo_cate(cate_feat)
            
            if eval:
                cate_pred=points_nms(cate_pred.sigmoid(),kernel=2).permute(0,2,3,1)
            return cate_pred,kernel_pred
        
        def loss(self,cate_pred,kernel_pred,ins_pred,
                 gt_bbox_list,gt_label_list,gt_mask_list,
                 img_metas,cfg,gt_bboxes_ignore=None):
            mask_feat_size=ins_pred.size()[-2:]
            ins_label_list,cate_label_list,ins_ind_label_list,grid_order_list=multi_apply(
                self.solov2_target_single,
                gt_bbox_list,
                gt_label_list,
                gt_mask_list,
                mask_feat_size=mask_feat_size
                )
            ins_labels=[torch.cat([ins_label_level_img for ins_label_level_img in ins_label_level],0)
                        for ins_label_level in zip(*ins_label_list)]
            kernel_preds=[[kernel_preds_level_img.view(kernel_preds_level_img.shape[0],-1)[:,grid_orders_level_img]
               for kernel_preds_level_img,grid_orders_level_img in zip(kernel_preds_level,grid_orders_level)]
                          for kernel_preds_level,grid_orders_level in zip(kernel_pred,zip(*grid_order_list))]
            
            #Generate mask
            ins_pred=ins_pred
            ins_pred_list=[]
            for b_kernel_pred in kernel_pred:
                b_mask_pred=[]
                for idx,kernel_pred in enumerate(b_kernel_pred):
                    if kernel_pred.size()[-1]==0:
                        continue
                    cur_ins_pred=ins_pred[idx,...]
                    H,W=cur_ins_pred.shape[-2:]
                    N,I=kernel_pred.shape
                    cur_ins_pred=cur_ins_pred.unsqueeze(0)
                    kernel_pred=kernel_pred.permute(1,0).view(1,-1,1,1)
                    cur_ins_pred=F.conv2d(cur_ins_pred,kernel_pred,stride=1).view(-1,H,W)
                    b_mask_pred.append(cur_ins_pred)
                if len(b_mask_pred)==0:
                    b_mask_pred=None
                else:
                    b_mask_pred=torch.cat(b_mask_pred,0)
                ins_pred_list.append(b_mask_pred)
            ins_ind_labels=[
                torch.cat([ins_ind_labels_level_img.flatten()
                           for ins_ind_labels_level_img in ins_ind_labels_level])
                for ins_ind_labels_level in zip(*ins_ind_label_list)
                ]
            flatten_ins_ind_labels=torch.cat(ins_ind_labels)
            num_ins=flatten_ins_ind_labels.sum()
            
            #dice loss
            loss_ins=[]
            for inp,target in zip(ins_pred_list,ins_labels):
                if inp is None:
                    continue
                inp=torch.sigmoid(inp)
                loss_ins.append(dice_loss(inp,target))
            loss_ins=torch.cat(loss_ins).mean()
            loss_ins=loss_ins*self.ins_loss_weight
            #cate
            cate_labels=[
                torch.cat([cate_labels_level_img.flatten()
                           for cate_labels_level_img in cate_labels_level])
                for cate_labels_level in zip(*cate_label_list)
                ]
            flatten_cate_label=torch.cat(cate_labels)
            cate_preds=[
                cate_pred.permute(0,2,3,1).reshape(-1,self.cate_out_channel)
                for cate_pred in cate_pred
                ]
            flatten_cate_pred=torch.cat(cate_preds)
            loss_cate=self.loss_cate(flatten_cate_pred,flatten_cate_label,avg_factor=num_ins+1)
            return dict(loss_ins=loss_ins,
                        loss_cate=loss_cate)
        
        def solov2_target_single(self,gt_bboxes_raw,
                                 gt_labels_raw,gt_masks_raw,
                                 mask_feat_size):
            device=gt_labels_raw[0].device
            #INS
            gt_areas=torch.sqrt((gt_bboxes_raw[:,2]-gt_bboxes_raw[:,0])*
                                (gt_bboxes_raw[:,3]-gt_bboxes_raw[:,1]))
            ins_label_list=[]
            cate_label_list=[]
            ins_ind_label_list=[]
            grid_order_list=[]
            
            for (lower_bound,upper_bound),stride,num_grid\
                in zip(self.scale_ranges,self.strides,self.seg_num_grids):
                    hit_indices=((gt_areas>=lower_bound)&(gt_areas<=upper_bound)).nonzero().flatten()
                    num_ins=len(hit_indices)
                    ins_label=[]
                    grid_order=[]
                    cate_label=torch.zeros([num_grid,num_grid],dtype=torch.int64,device=device)
                    ins_ind_label=torch.zeros([num_grid**2],dtype=torch.bool,device=device)
                    
                    if num_ins==0:
                        ins_label=torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                        ins_label_list.append(ins_label)
                        cate_label_list.append(cate_label)
                        ins_ind_label_list.append(ins_ind_label)
                        grid_order_list.append([])
                        continue
                    gt_bboxes=gt_bboxes_raw[hit_indices]
                    gt_labels=gt_labels_raw[hit_indices]
                    gt_masks=gt_masks_raw[hit_indices]
                    
                    half_ws=0.5*(gt_bboxes[:,2]-gt_bboxes[:,0])*self.sigma
                    half_hs=0.5*(gt_bboxes[:,3]-gt_bboxes[:,1])*self.sigma
                    
                    #mask_cente
                    gt_mask_pt=torch.from_numpy(gt_masks).to(device)
                    center_ws,center_hs=center_of_mass(gt_mask_pt)
                    valid_mask_flag=gt_mask_pt.sum(dim=-1).sum(dim=-1)>0
                    
                    output_stride=4
                    for  seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flag):
                        if not valid_mask_flag:
                            continue
                        upsampled_size=(mask_feat_size[0]*4,mask_feat_size[1]*4)
                        coord_w=int((center_w/upsampled_size[1])//(1./num_grid))
                        coord_h=int((center_h/upsampled_size[0])//(1./num_grid))
                        #L,T,R,D
                        top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                        down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                        left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                        right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                        top = max(top_box, coord_h-1)
                        down = min(down_box, coord_h+1)
                        left = max(coord_w-1, left_box)
                        right = min(right_box, coord_w+1)
                        
                        cate_label[top:(down+1),left:(right+1)]=gt_label
                        seg_mask=mmcv.imrescale(seg_mask,scale=1./output_stride)
                        seg_mask=torch.from_numpy(seg_mask).to(device=device)
                        
                        for i in range(top,down+1):
                            for j in range(left,right+1):
                                label=int(i*num_grid+j)
                                cur_ins_label=torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                                cur_ins_label[:seg_mask.shape[0],:seg_mask.shape[1]]=seg_mask
                                ins_label.append(cur_ins_label)
                                ins_ind_label[label]=True
                                grid_order.append(label)
                    if len(ins_label)==0:
                        ins_label=torch.zeros([0,mask_feat_size[0],mask_feat_size[1]],dtype=torch.uint8,device=device)
                    else:
                        ins_label=torch.stack(ins_label,0)
                    ins_label_list.append(ins_label)
                    cate_label_list.append(cate_label)
                    ins_ind_label_list.append(ins_ind_label)
                    grid_order_list.append(grid_order)
            return ins_label_list,cate_label_list,ins_ind_label_list,grid_order_list
        
        def get_seg(self,cate_preds,kernel_preds,seg_pred,img_metas,cfg,rescale=None):
            
            num_levels=len(cate_preds)
            featmap_size=seg_pred.size()[-2:]
            
            result_list=[]
            for img_id in range(len(img_metas)):
                cate_pred_list=[
                    cate_preds[i][img_id].view(-1,self.cate_out_channels).detach() for i in range(num_levels)
                    ]
                seg_pred_list=seg_pred[img_id,...].unsqueeze(0)
                kernel_pred_list=[
                    kernel_preds[i][img_id].permute(1,2,0).view(-1,self.kernel_out_channel).detach() for i in range(num_levels)
                    ]
                
                img_shape=img_metas[img_id]['img_shape']
                scale_factor=img_metas[img_id]['scale_factor']
                ori_shape=img_metas[img_id]['ori_shape']
                
                cate_pred_list=torch.cat(cate_pred_list,dim=0)
                kernel_pred_list=torch.cat(kernel_pred_list,dim=0)
                
                result=self.get_seg_single(cate_pred_list,seg_pred_list,kernel_pred_list,
                                           featmap_size,img_shape,ori_shape,scale_factor,cfg,rescale)
                result_list.append(result)
            return result_list
        
        def get_seg_single(self,cate_pred,kernel_pred,seg_pred,featmap_size,img_shape,ori_shape,scale_factor,cfg,rescale=False,debug=False):
            
            #overall info
            h,w,_=img_shape
            upsampled_size_out=(featmap_size[0]*4,featmap_size[1]*4)
            #process
            inds=(cate_pred>cfg.score_thr)
            cate_score=cate_pred[inds]
            if len(cate_score)==0:
                return None
            #cate label and kernel pred
            inds=inds.nonzero()
            cate_labels=inds[:,1]
            kernel_preds=kernel_pred[inds[:,0]]
            #trans
            size_trans=cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
            strides=kernel_pred.new_ones(size_trans[-1])
            
            n_stage=len(self.seg_num_grids)
            strides[:size_trans[0]]*=self.strides[0]
            
            for ind in range(1,n_stage):
                strides[size_trans[ind-1]:size_trans[ind]]*=self.strides[ind]
            strides=strides[inds[:,0]]
            
            #Mask encoding
            I,N=kernel_pred.shape
            kernel_pred=kernel_pred.view(I,N,1,1)
            seg_pred=F.conv2d(seg_pred,kernel_pred,stride=1).unsqueeze(0).sigmoid()
            seg_mask=seg_pred>cfg.mask_thr
            sum_mask=seg_mask.sum((1,2)).float()
            #filter
            keep=sum_mask>strides
            if keep.sum()==0:
                return None
            seg_mask=seg_mask[keep,...]
            seg_pred=seg_pred[keep,...]
            sum_mask=sum_mask[keep]
            cate_score=cate_score[keep]
            cate_labels=cate_labels[keep]
            
            seg_score=(seg_pred*seg_mask.float()).sum((1,2))/sum_mask
            cate_score*=seg_score
            
            #Apply matrix NMS
            sort_ind=torch.argsort(cate_score,descending=True)
            if len(sort_ind)>cfg.nms_pre:
                sort_ind=sort_ind[:cfg.nms_pre]
            seg_mask=seg_mask[sort_ind,:,:]
            seg_pred=seg_pred[sort_ind,:,:]
            sum_mask=sum_mask[sort_ind]
            cate_score=cate_score[sort_ind]
            cate_labels=cate_labels[sort_ind]
            
            cate_score=matrix_nms(seg_mask,cate_labels,cate_score,
                                  kernel=cfg.kernel,sigma=cfg.sigma,sum_masks=sum_mask)
            
            #Final filter and then sort and keep top x
            
            sort_inds=torch.argsort(cate_score,descending=True)
            if len(sort_inds)>cfg.max_per_img:
                sort_inds=sort_inds[:cfg.max_per_img]
            seg_pred=seg_pred[sort_inds,:,:]
            cate_score=cate_score[sort_inds]
            cate_labels=cate_labels[sort_inds]
            
            seg_pred=F.interpolate(seg_pred.unsqueeze(0),size=upsampled_size_out,mode='bilinear')[:,:,:h,:w]
            seg_masks=F.interpolate(seg_pred,size=ori_shape[:2],mode='bilinear').unsqueeze(0)
            seg_masks=seg_masks>cfg.mask_thr
            return seg_masks,cate_labels,cate_score
        
        
            
                
                        
            
            
            
        
        
        
    
    