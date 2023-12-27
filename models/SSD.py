# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .base.roialign import ROIAlign
import torch.nn.functional as F
from torchvision.models import resnet, vgg

from .base.feature import extract_feat_res, extract_feat_vgg
from .base.similarity import Similarity
from .learner import Learner

from functools import reduce
from operator import add
from .base.FEM import FEM


class SSD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roi = 16
        self.roialign = [ROIAlign(self.roi, 1./8), ROIAlign(self.roi, 1./16), ROIAlign(self.roi, 1./32)]

        # 1. Backbone network initialization
        self.backbone_type = 'resnet50'
        self.shot = config.DATA.SHOT
        self.gamma = config.MODEL.GAMMA
        self.eta = config.MODEL.ETA

        if self.backbone_type == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.roi_l = [3, 6, 7]
        elif self.backbone_type == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.roi_l = [4, 10, 13]
        elif self.backbone_type == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
            self.roi_l = [4, 27, 30]
        else:
            raise Exception('Unavailable backbone: %s' % self.backbone_type)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.learner = Learner(list(reversed(nbottlenecks[-3:])))

        dims = [512, 1024, 2048]
        vgg_dims = [512, 512, 512]
        if self.backbone_type == 'vgg16':
            self.fem = [FEM(vgg_dims[i]).cuda() for i in range(3)]
        else:
            self.fem = [FEM(dims[i]).cuda() for i in range(3)]

    def forward(self, image, boxes, mode):
        flag = False
        b, _, imh, imw = image.shape
        if mode == 'car_test':
            shot = 12
            with torch.no_grad():
                query_feats = self.extract_feats(image, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = []
            for idx in range(shot):
                support_feats.append([])
            roi_l = 0
            for idx, (query_feat, support_feat) in enumerate(zip(query_feats, boxes)):
                if idx >= self.roi_l[roi_l]:
                    roi_l += 1
                _, shot, c, sh, sw = support_feat.shape
                query_feats[idx], support_feat = self.fem[roi_l](query_feat, support_feat)
                support_feat = support_feat.view(1, shot, c, sh, sw).transpose(0, 1)
                for i in range(shot):
                    support_feats[i].append(support_feat[i])
        else:
            b_size = torch.stack((boxes[:, 4] - boxes[:, 2], boxes[:, 3] - boxes[:, 1]), dim=-1)
            bs_mean = b_size.view(-1, self.shot, 2).float().mean(dim=1, keepdim=False)
            if torch.any(bs_mean < self.gamma):
                flag = True
                min_scale = bs_mean.min()
                expand_scale = (self.gamma - min_scale) / self.eta + 1
                if expand_scale > 3:
                    expand_scale = 3
                if mode == 'train' and expand_scale > 2:
                    expand_scale = 2
                image = F.interpolate(image, (int(imh * expand_scale), int(imw * expand_scale)), mode='bilinear')
                boxes[:, 1: 5] = boxes[:, 1: 5] * expand_scale
            with torch.no_grad():
                query_feats = self.extract_feats(image, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = []
            for idx in range(self.shot):
                support_feats.append([])
            roi_l = 0
            for idx, query_feat in enumerate(query_feats):
                if idx >= self.roi_l[roi_l]:
                    roi_l += 1
                support_feat = self.roialign[roi_l](query_feat, boxes)
                _, c, sh, sw = support_feat.shape
                support_feat = support_feat.view(b, self.shot, c, sh, sw)
                query_feats[idx], support_feat = self.fem[roi_l](query_feat, support_feat)
                support_feat = support_feat.view(b, self.shot, c, sh, sw).transpose(0, 1)
                for i in range(self.shot):
                    support_feats[i].append(support_feat[i])
        pred_den = []
        for idx in range(self.shot):
            sim = Similarity.multilayer_similarity(query_feats, support_feats[idx], self.stack_ids)
            pred_den.append(self.learner(sim))
        pred_den = torch.cat(pred_den, dim=1).mean(dim=1, keepdim=True)

        if flag:
            original_sum = pred_den.sum(dim=(1, 2, 3), keepdim=True)
            pred_den = F.interpolate(pred_den, (imh, imw), mode='bilinear')
            pred_den = pred_den / pred_den.sum(dim=(1, 2, 3), keepdim=True) * original_sum
        return pred_den

    