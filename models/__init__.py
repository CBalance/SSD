# -*- coding: utf-8 -*-

from .SSD import SSD
from .loss.genloss import GeneralizedLoss
from .loss.mseloss import MSELoss
        

def build_model(config):
    model = SSD(config)
    return model, GeneralizedLoss(config.MODEL.FACTOR)
