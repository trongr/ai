"""
Evolution simulation.
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # For weird matplotlib bug crashing on OSX
import matplotlib.pyplot as plt
import random


def printFitness(generation, fitness):
    print("Generation:", generation, "Fitness:", fitness)


class Individual(object):
    def __init__(self, genes=None, mutate_prob=0.01):
        """
        If an individual is initialized with genes, e.g. as when given birth by
        parents, then at birth (here) we mutate a random gene. If not, then
        we're at the beginning of time and we randomly initialize the genes.
        """
        if genes is None:
            self.genes = np.random.randint(101, size=10)
        else:
            self.genes = genes
            if mutate_prob > np.random.rand():
                mutate_index = np.random.randint(len(self.genes) - 1)
                self.genes[mutate_index] = np.random.randint(101)

    def fitness(self):
        """
        Returns fitness of individual. Fitness is the difference between the
        target and the sum of an individual's genes. NOTE that by this
        definition the lower the "fitness" the more fit the individual is! How
        odd!
        """
        TARGET_SUM = 900
        return abs(TARGET_SUM - np.sum(self.genes))


class Population(object):
    def __init__(self, pop_size=10, mutate_prob=0.01, retain=0.2, random_retain=0.03):
        """
        - pop_size: size of population
        - fitness_goal: goal that population will be graded against
        - mutate_prob: probability that a new born individual in the population
          will develop a random mutation.
        - retain: percentage of top individuals in a generation that will go on
          to the next generation.
        - random_retain: probability that an unfit individual (not one of the
          top retained) will also survive.
        """
        self.pop_size = pop_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.random_retain = random_retain
        self.individuals = []
        self.parents = []
        self.fitness_history = []
        self.done = False

        """
        Randomly initialize individuals in the population.
        """
        for x in range(pop_size):
            self.individuals.append(Individual(genes=None, mutate_prob=mutate_prob))

    def grade(self):
        """
        Calculate population fitness, defined as the average of the fitness of
        all individuals in the population. Also add the population fitness to
        fitness history.
        - int generation: the current generation.
        + int popFitness: population fitness.
        """
        fitnessSum = 0
        for x in self.individuals:
            fitnessSum += x.fitness()

        _popFitness = fitnessSum / self.pop_size
        self.fitness_history.append(_popFitness)

        if int(round(_popFitness)) == 0:
            self.done = True  # Set done true if we hit target

        return _popFitness

    def select_parents(self):
        """
        Select the fittest individuals to be the parents of next generation.
        Also select a some random non-fittest individuals to help get us out of
        local maximums. This is essentially adding randomness / noise, so the
        genes have a way of exploring the solution space. In biological terms,
        we're diversifying the gene pool: it's not good to in-breed royalties,
        no matter how good their genes are.
        """
        # Sort individuals by fitness We use reversed because in this case lower
        # fitness is better.
        self.individuals = list(
            reversed(sorted(self.individuals, key=lambda x: x.fitness(), reverse=True))
        )

        # Keep the fittest as parents for next gen.
        retain_length = self.retain * len(self.individuals)
        self.parents = self.individuals[: int(retain_length)]

        # Randomly select some unfit individuals and let them survive into next
        # gen. Wouldn't misfits have been a better name? LOL
        unfittest = self.individuals[int(retain_length) :]
        for unfit in unfittest:
            if self.random_retain > np.random.rand():
                self.parents.append(unfit)

    def breed(self):
        """
        Crossover the parents to generate children and new generation of
        individuals.
        + [int] individuals: individuals in the population.
        """
        target_children_size = self.pop_size - len(self.parents)
        children = []
        if self.parents:
            while len(children) < target_children_size:
                father = random.choice(self.parents)
                mother = random.choice(self.parents)
                if father != mother:
                    child_genes = [
                        random.choice(pixel_pair)
                        for pixel_pair in zip(father.genes, mother.genes)
                    ]
                    child = Individual(child_genes)
                    children.append(child)
            self.individuals = self.parents + children

        return self.individuals


if __name__ == "__main__":
    pop_size = 1000
    mutate_prob = 0.01
    retain = 0.1
    random_retain = 0.03

    pop = Population(
        pop_size=pop_size,
        mutate_prob=mutate_prob,
        retain=retain,
        random_retain=random_retain,
    )

    SHOW_PLOT = True
    GENERATIONS = 5000
    for x in range(GENERATIONS):
        popFitness = pop.grade()
        pop.select_parents()
        pop.breed()

        # TODO. Visualize population

        if x % 5 == 0:
            printFitness(x, popFitness)
        elif pop.done:
            printFitness(x, popFitness)
            break

    if SHOW_PLOT:
        print("Showing fitness history graph")
        plt.plot(np.arange(len(pop.fitness_history)), pop.fitness_history)
        plt.ylabel("Fitness")
        plt.xlabel("Generations")
        plt.title(
            "Fitness - pop_size {} mutate_prob {} retain {} random_retain {}".format(
                pop_size, mutate_prob, retain, random_retain
            )
        )
        plt.show()
        plt.close()
