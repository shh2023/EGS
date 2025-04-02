import torch
from torch_scatter import scatter_add
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import numpy as np

def get_missing_feature_mask(rate, n_nodes, n_features, type="uniform"):

    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()


def get_mask(idx, num_nodes):

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_symmetrically_normalized_adjacency(edge_index, n_nodes):

    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD

def get_symmetrically_normalized_adjacency_l(edge_index, n_nodes):
    edge_weight = torch.ones((edge_index.size()[1],), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    

    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    

    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    
    
    DAD_values = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    DAD_indices = edge_index
    DAD_shape = torch.Size([n_nodes, n_nodes])
    
    DAD_sparse = torch.sparse.FloatTensor(DAD_indices, values=DAD_values, size=DAD_shape).coalesce()

    return DAD_sparse



