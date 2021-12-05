"""
Tugay Abdullazade

This project is implementation of generative algorithm solution to N Queens problem.
I have Chromosome class in a separate which has methods for placing queens randomly
on the board. I made an NxN board with 1s and 0s instead of creating a list with numbers
from 1 to N. I thought that would be more challenging and decided to do that. In this case,
at the start there can be some columns with multiple queens and some columns with no queens.
After crossover this led to some boards having more than n queens and some boards having less
queens and sometimes even no queen. To address this issue I modified the fitness function,
now it takes queen count into consideration and further punishes those boards which had different
number of queens than n. After that I realized that I can't use the "more is better" rule about
fitness function, because some boards had negative fitness value, so I inverted it and now it is
"less is better" and when we have 0 as fitness value then we have found a solution. Also I included
initial queen count as an extra feature because it converges to solution with different queen count.
One last thing, after a solution is found, it will be printed in console and a chessboard images
will be generated and saved at *result.png*

Note 1: I haven't handled all the edge cases, for example 10 queens and 101 queens at the start.
Note 2: Theoretically the code handles different combinations bu unfortunately sometimes it suffers
        from premature convergence and the fitness score doesn't improve from 1 or 2.
"""

# from typing import Tuple # if you don't have Python 3.9, then this line will raise error.
import numpy as np
from chromosome import Chromosome

# this variable is for determining which part of the generation will survive and produce new children
SURVIVAL_RATE = 0.8
# not all chromosomes mutate and to determine if a chromosome mutates, we compare random number with this variable
MUTATION_PROBABILITY = 0.03
# count of chromosomes in given population
POPULATION_SIZE = 2000
# count of queens over the board and the board size
QUEENS = 9
INITIAL_QUEENS = 10


class NQueens:
    def __init__(self, population_size: int, queens: int,
                 survival_rate: [float, int], mutation_probability: float, initial_queens: int) -> None:
        self.population_size = population_size
        self.board_size = queens
        self.mutation_probability = mutation_probability
        self.initial_queens = initial_queens
        self.parent_count = int(population_size * survival_rate)

        self.individuals = np.zeros(shape=(population_size, self.board_size, self.board_size), dtype=np.uint8)
        self.fitness_scores = np.zeros(shape=population_size)
        self.parents = np.zeros(shape=(self.parent_count, self.board_size, self.board_size), dtype=np.uint8)
        self.parent_scores = np.zeros(shape=(self.parent_count, self.board_size, self.board_size))

        self.generate_population()

    def calculate_fitness_scores(self) -> None:
        """This function calculates fitness scores of all chromosomes and stores int0 a variable."""
        self.fitness_scores = np.array([c.fitness_score() for c in self.individuals])

    def generate_population(self) -> None:
        """This function creates chromosomes for population"""
        self.individuals = np.array([Chromosome(n_queens=self.board_size,
                                                mutation_probability=self.mutation_probability,
                                                initial_queens=self.initial_queens) for _ in
                                     range(self.population_size)])

    def survive(self) -> None:
        """
        This function selects a part of the chromosomes based on their fitness scores to be
        parents for the next generation
        """
        best_indexes = self.fitness_scores.argsort()[:self.parent_count]
        self.parents = self.individuals.copy()[best_indexes]
        self.parent_scores = self.fitness_scores[best_indexes]

    def select_parents(self, parent_scores: np.ndarray):
        """
        This functions selects 2 different (based on index, not values) parents from all parents.
        Not all parents have the same chance of being selected. If some parent have more fitness
        score, then it have more chances of being selected
        """
        counts = np.random.multinomial(2, parent_scores)
        while len(np.where(counts > 0)[0]) != 2:
            counts = np.random.multinomial(2, parent_scores)
        parent1_index, parent2_index = np.where(counts > 0)[0]
        parent1 = self.parents[parent1_index]
        parent2 = self.parents[parent2_index]
        return parent1, parent2

    def reproduce_new_population(self):  # -> Tuple[Chromosome, ...]:
        """
        This function creates new population from survived chromosomes - now parents.
        Firstly it selects 2 parents, then creates 2 children from those 2 parents.
        Finally the new children are added to the new population.
        """
        new_population = []
        max_parent_score = max(self.parent_scores)
        parent_scores = max_parent_score + 1 - self.parent_scores
        parent_scores = np.asarray(parent_scores).astype('float64')
        parent_scores = parent_scores / parent_scores.sum()

        for i in range(self.population_size // 2):
            parent1, parent2 = self.select_parents(parent_scores)
            child1, child2 = self.recombine(parent1, parent2)

            child1.mutate()
            child2.mutate()

            new_population.append(child1)
            new_population.append(child2)

        self.individuals = np.array(new_population)

    def recombine(self, parent1: Chromosome, parent2: Chromosome):  # -> Tuple[Chromosome, ...]:
        """This function finds crossover point and creates 2 new Chromosomes"""
        crossover_point = np.random.randint(1, self.board_size - 1)
        return (Chromosome.from_2_parents(parent1, parent2, crossover_point),
                Chromosome.from_2_parents(parent2, parent1, crossover_point))


def ga():
    p = NQueens(population_size=POPULATION_SIZE, queens=QUEENS, survival_rate=SURVIVAL_RATE,
                mutation_probability=MUTATION_PROBABILITY, initial_queens=INITIAL_QUEENS)

    p.calculate_fitness_scores()
    p.survive()
    gen = 0
    while min(p.fitness_scores) != 0:
        p.reproduce_new_population()
        p.calculate_fitness_scores()
        gen += 1
        print(f'Generation {gen}\'s best score: {min(p.fitness_scores)}')
        p.survive()

    best_board = p.individuals[p.fitness_scores.argmin()]
    try:
        best_board.get_image().save('result.png')
        print('You can check image at result.png')
    except Exception as e:
        print("Couldn't save image")


if __name__ == '__main__':
    ga()
