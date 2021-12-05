# NQueens


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
