# -*- coding: utf-8 -*-
"""
Model definition
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class FcModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(FcModel, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class AttentionModel(nn.Module):
    def __init__(self, nfeat, n_rules, nhidden, nclass):
        super(AttentionModel, self).__init__()
        self.nclass = nclass
        self.fc1 = nn.Linear(nfeat, nhidden)
        self.fc2 = nn.Linear(nhidden, n_rules)

    def forward(self, x_lf, x_l):
        x = torch.cat((x_lf, x_l), 1)
        z = self.fc2(torch.tanh(self.fc1(x)))
        score = F.softmax(z, dim=1)
        mask = (x_lf >= 0).float()
        coverage_score = score * mask

        score_matrix = torch.empty(len(x_lf), self.nclass, device=x_lf.device)
        for k in range(self.nclass):
            score_matrix[:, k] = (score * (x_lf == k).float()).sum(dim=1)

        softmax_new_y = F.log_softmax(score_matrix, dim=1)
        return softmax_new_y, coverage_score


class AssembleModel(nn.Module):
    def __init__(self, n_features, n_rules, d_hid, n_class):
        super(AssembleModel, self).__init__()
        self.n_class = n_class
        self.fc_model = FcModel(n_features, d_hid, n_class)
        # self.fc_model2 = FcModel2(n_features + n_rules, d_hid, n_class) #TODO
        self.attention = AttentionModel(n_features + n_rules, n_rules, d_hid, n_class) #TODO

    def forward(self, x_l, x_u, x_lf_l, x_lf_u):

        predict_l = self.fc_model(x_l)
        predict_u = self.fc_model(x_u)

        lf_y_l, all_scores = self.attention(x_lf_l, x_l)
        fix_score = F.softmax(torch.mean(all_scores, dim=0), dim=0)  # use the average as the fixed score
        #print("fix_score", fix_score)

        lf_y_u = torch.zeros((x_lf_u.size(0), self.n_class), dtype=torch.float, device=x_l.device)
        for k in range(self.n_class):
            lf_y_u[:, k] = (fix_score.unsqueeze(0).repeat([x_lf_u.size(0), 1]) * (x_lf_u == k).float()).sum(dim=1)
        lf_y_u /= torch.sum(lf_y_u, dim=1).unsqueeze(1)
        lf_y_u[lf_y_u != lf_y_u] = 0  # handle the 'nan' (divided by 0) problem
        lf_y_u = F.log_softmax(lf_y_u, dim=1).detach()

        return predict_l, predict_u, lf_y_l, lf_y_u, fix_score.detach()