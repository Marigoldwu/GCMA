# -*- coding: utf-8 -*-
import math
import torch


def get_importance_of_edge(adj):
    adj[adj <= 0] = 0
    D = torch.diag(adj.sum(1)).float()
    Ie = torch.matmul(torch.matmul(D, adj), D)
    Ie_max = torch.max(Ie)
    Ie_min = torch.min(Ie)
    Ie = (Ie - Ie_min) / (Ie_max - Ie_min)
    return Ie, D


def get_importance_of_node(Imp_edge, D):
    Imp_node = torch.diag(Imp_edge.sum(1)) / D
    Imp_node[torch.isinf(Imp_node)] = 1
    Imp_node[torch.isnan(Imp_node)] = 0

    return Imp_node


def similarity(X):
    X = X / torch.norm(X, dim=-1, keepdim=True)
    X[torch.isinf(X)] = 1
    X[torch.isnan(X)] = 0
    S = torch.mm(X, X.t())
    return S


def get_similarity(X, adj, sin=False):
    # 计算余弦相似度矩阵S
    S = similarity(X)
    if sin:
        S = torch.sin((math.pi / 2) * S)
    S[S <= 0] = 0
    S.fill_diagonal_(0)
    S = torch.mul(S, adj)

    return S


def get_imp_structure(adj):
    adj.fill_diagonal_(0)
    imp_edge, D = get_importance_of_edge(adj)
    imp_node = get_importance_of_node(imp_edge, D)
    imp_structure = torch.sigmoid(torch.matmul(torch.matmul(imp_node, imp_edge), imp_node))
    imp_structure = imp_structure + torch.ones_like(adj).cuda()
    return imp_structure


def load_fuzzy_graph(S, adj, ftype="train"):
    S = torch.mul(S, adj)
    S.fill_diagonal_(0)
    S[S < 0] = 0
    
    if ftype == "train":
        imp_edge, _ = get_importance_of_edge(S)
        A = adj + imp_edge + S
    else:
        A = (adj + S) / 2

    A[A > 1] = 1
    A.fill_diagonal_(0)
    A = A + torch.eye(A.shape[0]).cuda()
    return A
