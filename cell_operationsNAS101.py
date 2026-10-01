
import torch
import torch.nn as nn

from genotypesNAS101 import EDGE2INDEX, NUM_INTERMEDIATE

OPS = {}

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

class MaxPool(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(MaxPool, self).__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

OPS['conv3x3-bn-relu'] = lambda C_in, C_out, stride, affine, ts: ReLUConvBN(C_in, C_out, 3, stride, 1, affine, ts)
OPS['conv1x1-bn-relu'] = lambda C_in, C_out, stride, affine, ts: ReLUConvBN(C_in, C_out, 1, stride, 0, affine, ts)
OPS['maxpool3x3'] = lambda C_in, C_out, stride, affine, ts: MaxPool(C_in, C_out, stride, affine, ts)

class NAS101SearchCell(nn.Module):
    

    def __init__(self, C, num_nodes=NUM_INTERMEDIATE, affine=False, track_running_stats=False):
        super(NAS101SearchCell, self).__init__()
        self.C = C
        self.out_dim = C          
        self.num_nodes = num_nodes
        self._ops = nn.ModuleList()
        for _ in range(num_nodes):
            node_ops = nn.ModuleList()
            for op_name in ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']:
                node_ops.append(OPS[op_name](C, C, 1, affine, track_running_stats))
            self._ops.append(node_ops)

    def forward(self, x, alphas_ops, alphas_edges):
        
        edge_w = alphas_edges.view(-1)          
        states = [x]                            
        for i in range(1, self.num_nodes + 1):
            feats, ws = [], []
            for j in range(i):
                feats.append(states[j])
                ws.append(edge_w[EDGE2INDEX[(j, i)]])
            wstack = torch.stack(ws)
            wnorm = wstack / (wstack.sum() + 1e-8)
            h = sum(w * f for w, f in zip(wnorm, feats))
            opw = alphas_ops[i - 1]             
            out = sum(w * op(h) for w, op in zip(opw, self._ops[i - 1]))
            states.append(out)
        feats, ws = [], []
        for i in range(1, self.num_nodes + 1):
            feats.append(states[i])
            ws.append(edge_w[EDGE2INDEX[(i, 6)]])
        wstack = torch.stack(ws)
        wnorm = wstack / (wstack.sum() + 1e-8)
        return sum(w * f for w, f in zip(wnorm, feats))
