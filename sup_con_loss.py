import torch
import torch.nn as nn
import torch.nn.functional as F

def get_multiview_fromto_contrast_feats(n_feat1s, n_spatial_feat1s, n_feat2s, n_spatial_feat2s, labels_t1, labels_t2, num_classes, fusion='concat'):
    multi_view_fromto_n_feats_list = []
    fromto_labels_list = []

    for i, (n_feat1, n_spatial_feat1) in enumerate(zip(n_feat1s, n_spatial_feat1s)):
        n_feat1_repeat = n_feat1.repeat((n_feat2s.shape[0], 1))
        n_spatial_feat1_repeat = n_spatial_feat1.repeat((n_spatial_feat2s.shape[0], 1))
        fromto_n_feat = torch.cat([n_feat1_repeat, n_feat2s], dim=1)
        fromto_spatial_feat = torch.cat([n_spatial_feat1_repeat, n_spatial_feat2s], dim=1)
        multi_view_fromto_n_feat = torch.stack([fromto_n_feat, fromto_spatial_feat], dim=1)
        fromto_label = labels_t1[i].unsqueeze(0) * num_classes + labels_t2
        multi_view_fromto_n_feats_list.append(multi_view_fromto_n_feat)
        fromto_labels_list.append(fromto_label)
    multi_view_fromto_features = torch.cat(multi_view_fromto_n_feats_list, dim=0)
    fromto_labels = torch.cat(fromto_labels_list, dim=0)
    return multi_view_fromto_features, fromto_labels



class SCCLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, params_dict):
        super(SCCLoss, self).__init__()
        # params_dict['num_classes']
        self.temperature = params_dict['temperature']
        self.contrast_mode = params_dict['contrast_mode']
        self.base_temperature = params_dict['base_temperature']
        self.params_dict = params_dict

    def forward(self, feats_dict, labels_t1, labels_t2, y1s_hat=None, y2s_hat=None):
        fusion = 'concat'
        if 'fusion' in self.params_dict:
            fusion = self.params_dict['fusion']
        n_feat1s, n_spatial_feat1s = feats_dict['n_feat1'], feats_dict['n_spatial_feat1']
        n_feat2s, n_spatial_feat2s = feats_dict['n_feat2'], feats_dict['n_spatial_feat2']
        label_locs = {}
        for label in torch.cat([labels_t1.unique(), labels_t2.unique()]).unique():
            label_t1_locs = torch.where(labels_t1 == label)[0]
            label_t2_locs = torch.where(labels_t2 == label)[0]
            if len(label_t2_locs) * len(label_t1_locs) > 0:
                label_locs[label.item()] = [label_t1_locs, label_t2_locs]

        multi_view_fromto_features, fromto_labels = get_multiview_fromto_contrast_feats(n_feat1s, n_spatial_feat1s, n_feat2s, n_spatial_feat2s, labels_t1, labels_t2, self.params_dict['num_classes'], fusion=fusion)
        loss = self._contrast(multi_view_fromto_features, fromto_labels)

        return loss


    def _contrast(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss