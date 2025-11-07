from dataclasses import dataclass
from typing import List, Union

@dataclass
class Solution:
    """
    A dataclass to represent a single feasible solution (tour) for the TSP.

    Attributes:
    - sequence: The sequence of node indices representing the tour.
                Example: [0, 4, 1, 2, 3] for N=5.
    - fitness: The calculated total cost (sum of costs along the sequence).
               The objective is to minimize this value.
    """
    sequence: List[int]
    fitness: Union[float, int]  # Fitness can be positive or negative (for r2_* instances)

    def __lt__(self, other):
        """
        Defines the 'less than' comparison operator (<) based on fitness.
        This is crucial for selection processes (e.g., Hill Climber, ES)
        where the goal is MINIMIZATION (lower fitness is better).
        """
        return self.fitness < other.fitness

    def __eq__(self, other):
        """
        Defines the 'equals' comparison operator (==) based on fitness.
        """
        return self.fitness == other.fitness

# --- Optional Additions (If needed, based on the first README) ---
# The first README mentions a 'SolutionResults' object for storing benchmark outputs.
# If you need to store results history, you might add a second dataclass:

@dataclass
class SolutionResults:
    """
    A dataclass to store the results and history of a single solver run.
    This structure is used for the compressed storage format.
    """
    best_solution: Solution
    best_fitness: Union[float, int]
    best_sequence: List[int]
    history: dict  # Fitness evolution stored with RLE (e.g., {'values': [...], 'counts': [...]})
    params: dict   # Solver parameters used for this run
    # ... potentially other fields like best_history