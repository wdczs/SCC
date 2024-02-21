# -*- coding: utf-8 -*-

import torch
from typing import Dict

from poolings.module_base import ModuleBase


class GAP(ModuleBase):
    """
    Global average pooling.
    """
    default_hyper_params = dict()

    def __init__(self, hps=None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        self.first_show = True
        super(GAP, self).__init__(hps)
        self.exp = 1

        if hps is not None:
            self.exp = hps.EXP if 'EXP' in hps else 1

    def __call__(self, fea):
        fea = fea.mean(dim=3).mean(dim=2)
        fea = fea.pow(self.exp)

        return fea
