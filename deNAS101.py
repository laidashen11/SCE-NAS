
import numpy as np
import torch
from chromosomesNAS101 import chromosome
from populationNAS101 import Population

class DifferentialEvolution:
    def __init__(self, pop_size, tournament_size, device, num_nodes, num_ops, num_edges, mrate=0.05):
        self._device = device
        self._pop_size = pop_size
        self._tournament_size = tournament_size
        self.mutation_rate = mrate
        self._num_nodes = num_nodes
        self._num_ops = num_ops
        self._num_edges = num_edges

    @staticmethod
    def _same_genotype(ch, geno_key):
        k = ch.genotype_key()
        return torch.equal(k[0], geno_key[0]) and torch.equal(k[1], geno_key[1])

    def evolve(self, population):
        new_pop = Population(0, self._num_nodes, self._num_ops, self._num_edges, self._device)
        for i in range(self._pop_size):
            new_pop.get_population().append(population.get_population()[i])

        for i in range(population.get_population_size()):
            xi = population.get_population()[i]
            xi_geno = xi.genotype_key()
            for _ in range(20):
                candidates = np.random.choice(population.get_population_size(), size=3, replace=False)
                a = population.get_population()[candidates[0]]
                b = population.get_population()[candidates[1]]
                c = population.get_population()[candidates[2]]
                if (not self._same_genotype(b, xi_geno)) and (not self._same_genotype(c, xi_geno)):
                    break

            mutant = chromosome(self._num_nodes, self._num_ops, self._num_edges, self._device)
            mutant_factor = 0.8

            for chrom1, chrom2, chrom3, chrom4, chrom5, chrom6 in zip(
                    population.get_population()[0].arch_parameters,   
                    a.arch_parameters,                                 
                    b.arch_parameters,                                 
                    c.arch_parameters,                                 
                    xi.arch_parameters,                                
                    mutant.arch_parameters):                           
                for j in range(chrom1.shape[0]):
                    chrom6[j].data.copy_(chrom5[j]
                                         + (1 - mutant_factor) * (chrom1[j] - chrom5[j])
                                         + mutant_factor * (chrom3[j] - chrom4[j]))

            cross_chrom = chromosome(self._num_nodes, self._num_ops, self._num_edges, self._device)
            for chrom1, chrom2, chrom3 in zip(mutant.arch_parameters, xi.arch_parameters, cross_chrom.arch_parameters):
                rand_j = np.random.randint(0, chrom1.shape[0])
                for j in range(chrom1.shape[0]):
                    if np.random.rand() >= 0.5 or j == rand_j:
                        chrom3[j].data.copy_(chrom1[j].data)
                    else:
                        chrom3[j].data.copy_(chrom2[j].data)

            cross_chrom.set_mutate_factor(mutant_factor)
            new_pop.get_population().append(cross_chrom)

        return new_pop
