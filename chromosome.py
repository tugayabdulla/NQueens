import numpy as np
from PIL import Image
import os

class Chromosome:
    def __init__(self, n_queens: int, mutation_probability: float, initial_queens: int = 1, board: np.ndarray = None):
        '''


        :param n_queens: Count of queens and board size. must be a positive integer
        :param mutation_probability: Chances of this chromosome being mutated. Positive number between 0 and 1
        :param initial_queens: Chromosomes with more or less queens than intended count of queens don't survive
                               So I thought that it would be interesting to add initial queen count as parameter

        :param board: Chromosome can be made from give NxN array. This parameter is used when a chromosome is
                      created from crossover of parents
        '''

        self.board_size = n_queens
        self.mutation_probability = mutation_probability
        self.initial_queens = initial_queens
        self.board = board if board is not None else np.zeros((self.board_size, self.board_size))
        if board is None:
            self.generate()

    def generate(self):
        """Populates NxN board randomly with `initial_queen` queens"""
        pop_random = np.random.choice(self.board_size * self.board_size, self.initial_queens, replace=False)
        for ind in range(self.initial_queens):
            r = pop_random[ind]
            x, y = r // self.board_size, r % self.board_size
            self.board[y][x] = 1

    def perm_count(self, x: [int, float, np.ndarray]):
        """Returns count of permutations of given number or elements in numpy array"""
        return x * (x - 1) // 2

    def mutate(self):
        """
        This function controlls mutation. If the generated random number is less than mutation
        probability, then randomly selected 1 queen on the board moves to randomly selected empty
        square
        """
        if np.random.random() > self.mutation_probability: return
        flat_board = self.board.flatten()
        # When there is no queen on the board, then the chromosome adds 1 queen to random place
        # on the board
        if sum(flat_board) == 0:
            new_position = np.random.choice(np.where(flat_board == 0)[0])
            flat_board[new_position] = 1
        else:
            previous_position = np.random.choice(np.where(flat_board == 1)[0])
            new_position = np.random.choice(np.where(flat_board == 0)[0])
            flat_board[previous_position] = 0
            flat_board[new_position] = 1
        self.board = flat_board.reshape(self.board_size, self.board_size)

    def fitness_score(self) -> int:
        """
    Calculates fitness score of a chromosome by finding count of queens in same horizontal line,
    vertical line and diagonals. After that applies `perm_count` to each row, column etc. to get
    count of conflicts. Sometimes, there are more or less queens on the board than `board_size`,
    in these cases, `board_size` times difference count is added to count of conflicts to further
    decrease chances of this chromosome to survive

    Returns:
         int: fitness score
        """
        horizontal_conflicts = self.perm_count(self.board.sum(axis=1)).sum()
        vertical_conflicts = self.perm_count(self.board.sum(axis=0)).sum()

        queens_in_l_to_r_diagonals = [np.trace(self.board, offset=i) for i in
                                      range(-self.board_size + 1, self.board_size)]
        l_to_r_diagonal_conflicts = self.perm_count(np.array(queens_in_l_to_r_diagonals)).sum()

        flipped_board = np.fliplr(self.board)
        queens_in_r_to_l_diagonals = [np.trace(flipped_board, offset=i) for i in
                                      range(-self.board_size + 1, self.board_size)]
        r_to_l_diagonal_conflicts = self.perm_count(np.array(queens_in_r_to_l_diagonals)).sum()

        total_conflicts = horizontal_conflicts + vertical_conflicts + r_to_l_diagonal_conflicts + l_to_r_diagonal_conflicts
        queen_count_diff = abs(self.board_size - self.board.sum())
        return int(total_conflicts + queen_count_diff * self.board_size)

    @staticmethod
    def from_2_parents(parent1: 'Chromosome', parent2: 'Chromosome', crossover_point):
        """
        This function creates new Chromosome from 2 parents and given crossover point. Crossover is
        done on 1 dimension.
    Args:
        parent1 : The first parent
        parent2 (Chromosome): The second parent
        crossover_point (int): The point where we split parents
    Returns:
        Chromosome: Created child
        """
        child = parent1.board.copy()
        child[crossover_point:] = parent2.board[crossover_point:]
        return Chromosome(board=child, n_queens=parent1.board_size, mutation_probability=parent1.mutation_probability)

    def get_image(self):
        """
        Creates image of chessboard and queens on it.

        :return: PIL.Image
        """
        white = Image.open(os.path.join('resources', 'white.png'))
        black = Image.open(os.path.join('resources', 'black.png'))
        queen = Image.open(os.path.join('resources', 'queen.png'))
        chessboard = Image.new('RGB', (self.board_size * 150,self.board_size * 150))
        copy = chessboard.copy()
        for i in range(self.board_size):
            for j in range(self.board_size):

                offset = (i * 150, j * 150)
                if (i + j) % 2 == 0:
                    copy.paste(white, offset)
                else:
                    copy.paste(black, offset)
                if self.board[i][j]:
                    offset = (15 + i * 150, 15 + j * 150)
                    copy.paste(queen, offset, queen)

        return copy

    def __str__(self):
        fitness_score = f'Fitness score: {self.fitness_score()}\n'
        return fitness_score + str(self.board).replace('1.', '♛').replace('0.', 'x') \
            .replace(' [', '').replace('[', '').replace(']', '')
