import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor


from utils import get_symmetrically_normalized_adjacency


class FeatureAndDetaPropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeatureAndDetaPropagation, self).__init__()
        self.num_iterations = num_iterations


    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor , lambda_1 , lambda_2) -> Tensor:

        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        deta = self.get_normaldeta_matrix(out, edge_index, n_nodes)
        eye_matrix = torch.eye(n_nodes,n_nodes)
        eye_matrix = eye_matrix.cuda()
        for i in range(int(self.num_iterations)):
            inv_matrix = torch.inverse((1 / (2 * lambda_1)) * deta + eye_matrix)
            out = torch.matmul(inv_matrix, out)
            deta = -1 / (4 * lambda_2) * torch.matmul(out, out.t()) + deta
            #out = -torch.matmul(deta, out)+out
            #deta= -torch.matmul(out, out.t())+deta

            #inv_matrix = torch.inverse((1 / (2 * lambda_1)) * deta + eye_matrix)

            #out = torch.matmul(inv_matrix, out)
            #deta = -1 / (4 * lambda_2) * torch.matmul(out, out.t()) + deta

            #out = -torch.matmul(deta, out)+out
            #deta = -1 / (4 * lambda_2) * torch.matmul(out, out.t()) + deta
            # adj_dynamic = self.struct_learner(out, edge_index)
            # deta_update = eye_matrix - adj_dynamic
            # deta = deta + deta_update

        return out

    def get_normaldeta_matrix(self, x, edge_index, n_nodes):

        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
        eye_matrix = torch.eye(n_nodes,n_nodes)
        eye_matrix =  eye_matrix.cuda()
        deta = eye_matrix - adj

        return deta
