
from chromosomesNAS101 import chromosome

class Population:
    def __init__(self, pop_size, num_nodes, num_ops, num_edges, device):
        self.population = []
        for _ in range(pop_size):
            self.population.append(chromosome(num_nodes, num_ops, num_edges, device))

    def get_population(self):
        return self.population

    def get_population_size(self):
        return len(self.population)

    def pop_pop(self, indices_to_pop):
        for index in sorted(indices_to_pop, reverse=True):
            self.population.pop(index)
        return self.population

    def pop_sort(self):
        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)

    def set_population(self, new_population):
        self.population = new_population
