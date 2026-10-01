
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cell_operationsNAS101 import NAS101SearchCell as SearchCell
from cell_operationsNAS201 import ResNetBasicblock
from genotypesNAS101 import alphas_to_structure, NUM_INTERMEDIATE, NUM_OPS, NUM_EDGES, ALL_EDGES

class TinyNetwork(nn.Module):

    def __init__(self, C, N, num_classes, affine=False, track_running_stats=False, edge_relax='sparse'):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N
        self.edge_relax = edge_relax       
        self._edge_mask = None
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)      
            else:
                cell = SearchCell(C_prev, NUM_INTERMEDIATE, affine, track_running_stats)
            self.cells.append(cell)
            C_prev = cell.out_dim

        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self.arch_parameters = nn.Parameter(
            torch.rand(NUM_INTERMEDIATE, NUM_OPS), requires_grad=False)
        self.arch_parameters_edges = nn.Parameter(
            torch.rand(NUM_EDGES, 1), requires_grad=False)

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self):
        
        return [self.arch_parameters, self.arch_parameters_edges]

    def update_alphas(self, alphas_ops, alphas_edges):
        
        assert isinstance(alphas_ops, torch.Tensor) and isinstance(alphas_edges, torch.Tensor)
        assert alphas_ops.device == self.arch_parameters.device
        assert alphas_edges.device == self.arch_parameters_edges.device
        assert alphas_ops.size() == self.arch_parameters.size(), alphas_ops.size()
        assert alphas_edges.size() == self.arch_parameters_edges.size(), alphas_edges.size()
        self.arch_parameters.data.copy_(alphas_ops)
        self.arch_parameters_edges.data.copy_(alphas_edges)
        self._edge_mask = None          

    def discretize(self):
        
        with torch.no_grad():
            ops_sm = F.softmax(self.arch_parameters, dim=-1)
            ops_idx = ops_sm.max(-1, keepdim=True)[1]
            ops_onehot = torch.zeros_like(ops_sm).scatter_(-1, ops_idx, 1.0)
            edges_bin = (self.arch_parameters_edges > 0.5).float()
            self.arch_parameters.data.copy_(ops_onehot)
            self.arch_parameters_edges.data.copy_(edges_bin)
        self._edge_mask = None
        return ops_onehot, edges_bin

    def _get_edge_mask(self):
        
        if self._edge_mask is None:
            struct = alphas_to_structure(self.arch_parameters, self.arch_parameters_edges)
            m = np.zeros(NUM_EDGES, dtype=np.float32)
            for k, (s, d) in enumerate(ALL_EDGES):
                if struct.adj_matrix[s, d] == 1:
                    m[k] = 1.0
            self._edge_mask = torch.tensor(
                m, device=self.arch_parameters_edges.device,
                dtype=self.arch_parameters_edges.dtype).view(-1, 1)
        return self._edge_mask

    def check_alphas(self, alphas_ops, alphas_edges):
        assert isinstance(alphas_ops, torch.Tensor) and isinstance(alphas_edges, torch.Tensor)
        return (torch.all(self.arch_parameters == alphas_ops).item()
                and torch.all(self.arch_parameters_edges == alphas_edges).item())

    def genotype(self):
        
        return alphas_to_structure(self.arch_parameters, self.arch_parameters_edges)

    def forward(self, inputs):
        ops_sm = F.softmax(self.arch_parameters, dim=-1)          
        if self.edge_relax == 'sparse':
            edges_sm = self._get_edge_mask()                       
        else:
            edges_sm = torch.sigmoid(self.arch_parameters_edges)   

        feature = self.stem(inputs)
        for cell in self.cells:
            if isinstance(cell, SearchCell):
                feature = cell(feature, ops_sm, edges_sm)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits
