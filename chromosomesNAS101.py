
import torch
import random
import utils

class chromosome:
    def __init__(self, num_nodes, num_ops, num_edges, device):
        self._device = device
        self._num_nodes = num_nodes
        self._num_ops = num_ops
        self._num_edges = num_edges

        self.alphas_ops = torch.rand(num_nodes, num_ops, device=device)
        self.alphas_edges = torch.rand(num_edges, 1, device=device)
        self.arch_parameters = [self.alphas_ops, self.alphas_edges]

        self.objs = utils.AvgrageMeter()
        self.top1 = utils.AvgrageMeter()
        self.top5 = utils.AvgrageMeter()

        self.evaluated = False
        self.tmp = []
        self.mutate_factor = random.uniform(0.1, 0.9)

    def accumulate(self):
        self.tmp.append(self.top1.avg)

    def genotype_key(self):
        
        ops_idx = self.alphas_ops.argmax(dim=-1)
        edges_bin = (self.alphas_edges > 0.5).view(-1)
        return (ops_idx, edges_bin)

    def set_fitness(self, value, top1, top5):
        self.objs.avg = value
        self.top1.avg = top1
        self.top5.avg = top5

    def get_len(self):
        return self._num_nodes

    def get_fitness(self):
        return self.top1.avg

    def get_all_metrics(self):
        return self.objs, self.top1, self.top5

    def get_arch_parameters(self):
        return self.arch_parameters

    def get_mutate_factor(self):
        return self.mutate_factor

    def set_mutate_factor(self, mutate_factor):
        self.mutate_factor = mutate_factor
