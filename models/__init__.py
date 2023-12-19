# -*- coding: utf-8 -*-

from .hslnet import HSLNet
from .loss.genloss import GeneralizedLoss
from .loss.mseloss import MSELoss
        

def build_model(config):
    model = HSLNet(config)
    return model, GeneralizedLoss(config.MODEL.FACTOR)
