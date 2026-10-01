
import numpy as np

NAS_BENCH_101_OPS = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']

NUM_VERTICES = 7          
NUM_INTERMEDIATE = 5      
NUM_OPS = 3               

ALL_EDGES = []
for i in range(1, NUM_INTERMEDIATE + 1):
    ALL_EDGES.append((0, i))
for i in range(1, NUM_INTERMEDIATE + 1):
    for j in range(i + 1, NUM_INTERMEDIATE + 1):
        ALL_EDGES.append((i, j))
for i in range(1, NUM_INTERMEDIATE + 1):
    ALL_EDGES.append((i, 6))
NUM_EDGES = len(ALL_EDGES)   

EDGE2INDEX = {e: k for k, e in enumerate(ALL_EDGES)}

class Structure:
    

    def __init__(self, adj_matrix, ops):
        self.adj_matrix = np.asarray(adj_matrix, dtype=np.int32)
        self.ops = list(ops)
        assert self.adj_matrix.shape == (NUM_VERTICES, NUM_VERTICES), self.adj_matrix.shape
        assert len(self.ops) == NUM_INTERMEDIATE, len(self.ops)

    def tostr(self):
        
        ops_str = "|".join(self.ops)
        edges_str = ";".join(
            "{}->{}".format(s, d) for (s, d) in ALL_EDGES if self.adj_matrix[s, d] == 1)
        return "|" + ops_str + "|+|" + edges_str + "|"

    def __repr__(self):
        return self.tostr()

    def __eq__(self, other):
        return (isinstance(other, Structure)
                and np.array_equal(self.adj_matrix, other.adj_matrix)
                and self.ops == other.ops)

    def __hash__(self):
        return hash((self.adj_matrix.tobytes(), tuple(self.ops)))

def alphas_to_structure(alphas_ops, alphas_edges, threshold=0.5, max_edges=9):
    
    aops = np.asarray(alphas_ops.detach().cpu() if hasattr(alphas_ops, 'detach') else alphas_ops)
    aedg = np.asarray(alphas_edges.detach().cpu() if hasattr(alphas_edges, 'detach') else alphas_edges)
    aedg = aedg.reshape(-1)

    ops = [NAS_BENCH_101_OPS[int(np.argmax(aops[i]))] for i in range(NUM_INTERMEDIATE)]
    adj = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=np.int32)

    cand = sorted(range(NUM_EDGES), key=lambda k: aedg[k], reverse=True)
    cand = [k for k in cand if aedg[k] > threshold][:max_edges]
    for k in cand:
        s, d = ALL_EDGES[k]
        adj[s, d] = 1

    for i in range(1, NUM_INTERMEDIATE + 1):
        if adj[:, i].sum() == 0:
            adj[i - 1, i] = 1          
        if adj[i, :].sum() == 0:
            adj[i, i + 1] = 1          
    if adj[0, :].sum() == 0:
        adj[0, 1] = 1
    if adj[:, 6].sum() == 0:
        adj[NUM_INTERMEDIATE, 6] = 1

    while int(adj.sum()) > max_edges:
        removable = []
        for (s, d) in ALL_EDGES:
            if adj[s, d] == 0:
                continue
            d_ok = not (1 <= d <= NUM_INTERMEDIATE and adj[:, d].sum() <= 1)
            s_ok = not (1 <= s <= NUM_INTERMEDIATE and adj[s, :].sum() <= 1)
            if d_ok and s_ok:
                removable.append((s, d))
        if not removable:
            break
        drop = min(removable, key=lambda e: aedg[EDGE2INDEX[e]])
        adj[drop[0], drop[1]] = 0

    if int(adj.sum()) > max_edges:
        adj = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=np.int32)
        for i in range(NUM_INTERMEDIATE + 1):
            adj[i, i + 1] = 1

    return Structure(adj, ops)

def structure_to_alphas(structure):
    
    aops = np.zeros((NUM_INTERMEDIATE, NUM_OPS), dtype=np.float32)
    for i, op in enumerate(structure.ops):
        aops[i, NAS_BENCH_101_OPS.index(op)] = 1.0
    aedg = np.zeros((NUM_EDGES, 1), dtype=np.float32)
    for k, (s, d) in enumerate(ALL_EDGES):
        aedg[k, 0] = float(structure.adj_matrix[s, d])
    return aops, aedg
