import pdb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from segmentation_models_pytorch.encoders import get_encoder
import poolings as poolings
from sup_con_loss import SCCLoss

__all__ = ['SupConPoolPCSNet']


class SupConPoolPCSNet(nn.Module):
    def __init__(self, num_classes):
        super(SupConPoolPCSNet, self).__init__()
        self.seg_encoder = get_encoder(
            name='resnet50',
            in_channels=4,
            depth=5,
            weights=None,
        )

        proj_dim = 128
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )

        self.pooling = poolings.__dict__['GAP']()
        self.spatial_pooling = poolings.__dict__['GeM']()

        fc1 = []
        fc1.append(nn.Linear(2048, 1024, bias=True))
        fc1.append(nn.ReLU(True))
        fc1.append(nn.Linear(1024, 1024, bias=True))
        fc1.append(nn.ReLU(True))
        self.fc1 = nn.Sequential(*fc1)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x1, x2, labels=None):
        featmap1 = self.seg_encoder(x1)[-1]
        featmap2 = self.seg_encoder(x2)[-1]

        features1 = self.pooling(featmap1).reshape(x1.size(0), -1)
        features2 = self.pooling(featmap2).reshape(x1.size(0), -1)

        spatial_features1 = self.spatial_pooling(featmap1).reshape(x1.size(0), -1)
        spatial_features2 = self.spatial_pooling(featmap2).reshape(x1.size(0), -1)
        feat1 = self.fc1(features1)
        feat2 = self.fc1(features2)
        spatial_feat1 = self.fc1(spatial_features1)
        spatial_feat2 = self.fc1(spatial_features2)

        t1_logits = self.fc2(feat1)
        t2_logits = self.fc2(feat2)

        n_feat1 = F.normalize(self.head(feat1), dim=1)
        n_feat2 = F.normalize(self.head(feat2), dim=1)
        n_spatial_feat1 = F.normalize(self.head(spatial_feat1), dim=1)
        n_spatial_feat2 = F.normalize(self.head(spatial_feat2), dim=1)

        feat_dict = {
            'feat1': features1,
            'feat2': features2,
            'n_feat1': n_feat1,
            'n_feat2': n_feat2,
            'n_spatial_feat1': n_spatial_feat1,
            'n_spatial_feat2': n_spatial_feat2,
        }

        logits_dict = {
            't1_logits': t1_logits,
            't2_logits': t2_logits
        }

        return feat_dict, logits_dict

if __name__ == '__main__':
    num_classes = 9
    loss_ratio = 0.5
    params_dict = {
        'contrast_mode': 'all',
        'temperature': 0.1,
        'base_temperature': 0.1
    }
    params_dict['num_classes'] = num_classes
    SCC = SCCLoss(params_dict)
    cls_criterion = nn.CrossEntropyLoss()
    model = SupConPoolPCSNet(num_classes)

    img1 = torch.randn(32, 4, 150, 150)
    img2 = torch.randn(32, 4, 150, 150)
    label1 = torch.randint(8, (32,))
    label2 = torch.randint(8, (32,))
    feat_dict, logits_dict = model(img1, img2)
    scc_loss = SCC(feat_dict, label1, label2)
    t1_logits = logits_dict['t1_logits']
    t2_logits = logits_dict['t2_logits']
    t1_cls_loss = cls_criterion(t1_logits, label1)
    t2_cls_loss = cls_criterion(t2_logits, label2)
    loss = (t1_cls_loss + t2_cls_loss) / 2.0 + loss_ratio * scc_loss
    
    print(loss)