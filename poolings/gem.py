# -*- coding: utf-8 -*-
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from poolings.module_base import ModuleBase

class GeM(ModuleBase):
    """
    Generalized-mean pooling.
    c.f. https://pdfs.semanticscholar.org/a2ca/e0ed91d8a3298b3209fc7ea0a4248b914386.pdf

    Hyper-Params
        p (float): hyper-parameter for calculating generalized mean. If p = 1, GeM is equal to global average pooling, and
            if p = +infinity, GeM is equal to global max pooling.
    """
    default_hyper_params = {
        "p": 3.0,
        "trainable": False,
    }

    def __init__(self, hps=None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(GeM, self).__init__(hps)

        if self._hyper_params['trainable'] is True:
            self.p = nn.Parameter(torch.ones(1)*self._hyper_params["p"]).cuda()
        else:
            self.p = self._hyper_params["p"]
        self.exp = 1

        if hps is not None:
            self.exp = hps.EXP if 'EXP' in hps else 1

    def __call__(self, fea):
        p = self.p
        eps = 1e-6
        # pdb.set_trace()

        fea = fea.clamp(min=eps) ** p
        h, w = fea.shape[2:]
        fea = fea.sum(dim=(2, 3)) * 1.0 / w / h
        fea = fea ** (1.0 / p)
        # fea = F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
        # fea = fea.pow(self.exp)

        return fea
