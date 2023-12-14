# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model import BaseModule
from torch import Tensor

from SPGNet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
import torch
        
class DeformSpatialAttLayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(DeformSpatialAttLayer, self).__init__()
        self.spatial_conv = nn.Sequential(
            build_conv_layer(
                dict(type='DCNv2', deformable_groups=1),
                2,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size-1)//2,
            ),
            nn.BatchNorm2d(1)
        )


    def forward(self, x):
        # b, c, _, _ = x.size()
        y = torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]],dim=1)
        y = self.spatial_conv(y).sigmoid()
        # y = y.sigmoid()
        # return x * y.expand_as(x)
        # return y.expand_as(x)
        return y.expand_as(x)
    
    

class SpatialAttLayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttLayer, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2,1,kernel_size,stride=1,padding=(kernel_size-1) // 2),
            nn.BatchNorm2d(1)
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]],dim=1)
        y = self.spatial_conv(y).sigmoid()
        return x * y.expand_as(x)

class DynamicAttention(nn.Module):
    def __init__(self, channels, dynamic_heads=16):
        super(DynamicAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mid = dynamic_heads
        self.fc1 = nn.Linear(channels, channels * dynamic_heads, bias=False)
        self.norm1 = nn.LayerNorm(dynamic_heads)
        self.relu = nn.ReLU(inplace=True)
        
        """
        self.fc2 = nn.Linear(dynamic_heads, dynamic_heads * channels, bias=False)
        self.norm2 = nn.LayerNorm(channels)
        self.sig = nn.Sigmoid()
        """
        
        
        
        self.fc2 = nn.Sequential(
            nn.Linear(dynamic_heads, channels, bias=False),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        weight = self.fc1(y).view(b,c,self.mid)
        y = torch.matmul(y.view(b,1,c), weight).squeeze(1)
        y = self.relu(self.norm1(y))

        """
        weight = self.fc2(y).view(b,self.mid, c)
        y = torch.matmul(y.view(b,1,self.mid), weight).squeeze(1)
        y = self.sig(self.norm2(y)).view(b,c,1,1)
        """

        y = self.fc2(y).view(b,c,1,1)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class dynamic_eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3, dynamic_head=1):
        super(dynamic_eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, dynamic_head, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        # return y.expand_as(x)

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

"""
class SKDynamicChannelAttention(nn.Module):
    def __init__(self, channels, reduction, branches):
        super(SKDynamicChannelAttention, self).__init__()
        self.branches = branches
        self.mid = int(channels // reduction)
        self.convs = nn.ModuleList([])
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels,channels,kernel_size=3+2*i,stride=1,padding=1+i,groups=channels, bias=False),
                nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0, bias=False)
                # nn.BatchNorm2d(channels),
                # nn.ReLU()    
            ))
            self.fcs.append(nn.Sequential(
            nn.Linear(channels // reduction, channels, bias=False),
            # nn.Sigmoid()
        ))
            
        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels * channels // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        b,c,h,w = x.size()
        branches = [conv(x) for conv in self.convs]
        
        
        attn = torch.cat(branches,dim=1).view(b,self.branches,c,h,w).sum(1)
        attn = attn.mean(dim=[2,3])
        temp = self.fc1(attn).view(b,self.mid,c)
        attn = torch.matmul(temp,attn.view(b,c,1)).squeeze(2)
        
        atts = [fc(attn) for fc in self.fcs]
        atts = self.softmax(torch.cat(atts,dim=1).view(b,self.branches,c))
        atts = atts.split(1,dim=1)
        
        for i in range(self.branches):
            branches[i] = branches[i] * atts[i].view(b,c,1,1).expand(b,c,h,w)
        
        x = torch.cat(branches,dim=1).view(b,self.branches,c,h,w).sum(1)
        return x
"""

class SpatialSelectBlock(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialSelectBlock, self).__init__()
        # self.channel = int(channels // 2)
        # self.mid = int(channels // reduction)
        
        """
        self.residual = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
        )
        self.relu = nn.ReLU()
        """
        
        
        # self.conv1 = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        
        # self.conv2 = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        
        """
        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels * channels // reduction),
            nn.ReLU(),
        )
        """
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2,2,kernel_size=kernel_size,stride=1,padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(2)  
        )
        
        # self.fc2 = nn.Linear(channels // reduction, channels)
        # self.fc3 = nn.Linear(channels // reduction, channels, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x, y):
        b,c,h,w = x.size()
        # attn_x = torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]],dim=1) 
        # attn_y = torch.cat([y.mean(1,keepdim=True), y.max(1,keepdim=True)[0]],dim=1) 
        attn = torch.cat([x,y],dim=1)
        attn = torch.cat([attn.mean(1,keepdim=True),attn.max(1,keepdim=True)[0]], dim=1)
        # branches = [self.conv1(mid), self.conv2(mid)]
        # mid = torch.cat(branches,dim=1).view(b,2,c,h,w).sum(1)
        # mid = self.conv1(mid) + self.conv2(mid)
        
        # attn = mid.mean(dim=[2,3])
        # temp = self.fc1(attn).view(b,self.mid,c * 2)
        # attn = torch.matmul(temp,attn.view(b,c * 2,1)).squeeze(2)
        
        """
        att1 = self.fc2(attn)
        att2 = self.fc3(attn)
        atts = torch.cat([att1,att2],dim=1).sigmoid().view(b,2,c).split(1,dim=1)
        """
        atts = self.softmax(self.spatial_conv(attn)).split(1,dim=1)
        x = x * atts[0].expand(b,c,h,w)
        y = y * atts[1].expand(b,c,h,w)
        
        return x + y

class DynamicSelectBlock(nn.Module):
    def __init__(self, channels):
        super(DynamicSelectBlock, self).__init__()
        # self.channel = int(channels // 2)
        self.mid = int(channels // 16)
        
        """
        self.residual = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
        )
        self.relu = nn.ReLU()
        """
        
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0),
        )
        """

        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels * channels // 16, bias=False),
            nn.ReLU(),
        )

        """
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2,2,kernel_size=kernel_size,stride=1,padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(2)  
        )
        """
        
        self.fc2 = nn.Linear(channels // 16, channels // 2, bias=False)
        self.fc3 = nn.Linear(channels // 16, channels // 2, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x, y):
        b,c,h,w = x.size()
        # attn_x = torch.cat([x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]],dim=1) 
        # attn_y = torch.cat([y.mean(1,keepdim=True), y.max(1,keepdim=True)[0]],dim=1) 
        # attn = torch.cat([self.conv1(x),self.conv2(y)],dim=1)
        # attn = torch.cat([attn.mean(1,keepdim=True),attn.max(1,keepdim=True)[0]], dim=1)
        # branches = [self.conv1(mid), self.conv2(mid)]
        # mid = torch.cat(branches,dim=1).view(b,2,c,h,w).sum(1)
        # mid = self.conv1(mid) + self.conv2(mid)
        
        attn = torch.cat([x,y],dim=1).mean(dim=[2,3])
        temp = self.fc1(attn).view(b,self.mid,c * 2)
        attn = torch.matmul(temp,attn.view(b,c * 2,1)).squeeze(2)
        

        att1 = self.fc2(attn).sigmoid()
        att2 = self.fc3(attn).sigmoid()
        atts = self.softmax(torch.cat([att1,att2],dim=1)).view(b,2,c,1,1).split(1,dim=1)

        """
        atts = self.softmax(self.spatial_conv(attn)).split(1,dim=1)
        """

        x = x * atts[0].squeeze(1).expand(b,c,h,w)
        y = y * atts[1].squeeze(1).expand(b,c,h,w)
        
        return x + y
    
    
@MODELS.register_module()
class DynamicSelectedFPN(BaseModule):
    """Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        
        """
        self.eca_att0 = eca_layer(256,5)
        self.eca_att1 = eca_layer(256,5)
        self.eca_att = [self.eca_att0,self.eca_att1]
        """
        
        """
        self.se_att0 = SELayer(256)
        self.se_att1 = SELayer(256)
        self.se_att = [self.se_att0,self.se_att1]
        """
        
        
        
        """
        self.dyse_att0 = DynamicAttention(256, 32)
        self.dyse_att1 = DynamicAttention(256, 32)
        self.dyse_att = [self.dyse_att0,self.dyse_att1]
        """
        
        
        """
        self.spatial_att0 = SpatialAttLayer(7)
        self.spatial_att1 = SpatialAttLayer(7)
        self.spatial_att = [self.spatial_att0,
                            self.spatial_att1]
        """
        
        
        
        
        
        """
        self.despatial_att0 = DeformSpatialAttLayer(7)
        self.despatial_att1 = DeformSpatialAttLayer(7)
        self.despatial_att = [self.despatial_att0,
                            self.despatial_att1]
        """
        
        """
        self.dynamic_select0 = DynamicSelectBlock(512)
        self.dynamic_select1 = DynamicSelectBlock(512)
        self.dynamic_select = [
            self.dynamic_select0,
            self.dynamic_select1
        ]
        """
        
        
        self.SS_block0 = SpatialSelectBlock(11)
        self.SS_block1 = SpatialSelectBlock(11)
        self.SS_block = [
            self.SS_block0,
            self.SS_block1
        ]
    
        
        
        
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                """
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
                """
                
                

                
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = self.SS_block[i - 1](
                    laterals[i - 1], F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
                )
                
                
                
                # laterals[i - 1] = self.se_att[i - 1](laterals[i - 1])
                # laterals[i - 1] = self.spatial_att[i - 1](laterals[i - 1])
                # laterals[i - 1] = self.dyse_att[i - 1](laterals[i - 1])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
