# src/solvers/hc_solver.py

# Imports (assuming Solution and utils are available)
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_problems.solution import Solution

def epsilon_greedy_initialization(N, cost_matrix, p_random):
    """
    Initializes a feasible TSP tour using an epsilon-greedy nearest neighbor approach.
    N: Number of nodes/cities.
    cost_matrix: The distance/cost matrix.
    p_random: Probability of picking a random unvisited city instead of the nearest.
    """
    # Pseudocode based on README:
    # 1. Start sequence with a random_starting_index.
    # 2. Loop N-1 times:
    #    a. With probability (1 - p_random):
    #       Select the next index that MINIMIZES cost from the last node
    #       among those not already selected. (Nearest Neighbor)
    #    b. Else (with probability p_random):
    #       Select the next index RANDOMLY among those not already chosen.
    # 3. Add the selected index to the sequence.
    # 4. Calculate fitness and return a Solution object.
    
    # 1. Start sequence with a random_starting_index
    sequence = [np.random.randint(N)]
    unvisited = set(range(N)) - {sequence[0]}
    
    # 2. Loop N-1 times:
    for _ in range(N - 1):
        last_node = sequence[-1]
        
        #    a. With probability (1 - p_random):
        if np.random.random() < (1 - p_random):
            #       Select the next index that MINIMIZES cost from the last node
            #       among those not already selected. (Nearest Neighbor)
            next_node = min(unvisited, key=lambda node: cost_matrix[last_node][node])
        #    b. Else (with probability p_random):
        else:
            #       Select the next index RANDOMLY among those not already chosen.
            next_node = np.random.choice(list(unvisited))
        
        # 3. Add the selected index to the sequence.
        sequence.append(next_node)
        unvisited.remove(next_node)
    
    # 4. Calculate fitness and return a Solution object.
    return sequence


def two_opt_move(sequence, i, j):
    """
    Performs the 2-opt swap by reversing the segment between indices i and j.
    """
    # Reconnect edges (a-b, c-d) as (a-c, b-d) by reversing the middle segment.
    # sequence[i+1 : j+1] is reversed.
    new_sequence = sequence.copy()
    new_sequence[i+1:j+1] = reversed(sequence[i+1:j+1])
    return new_sequence

def hill_climber_solver(cost_matrix, max_iterations, initial_solution=None):
    """
    Runs the 2-opt Hill Climber.
    """
    # 1. Initialize with greedy_initialization if not provided.
    # 2. Loop until no improvement (or max_iterations/budget reached):
    #    a. Perform a 'sweep': Iterate through all possible 2-opt swaps (i, j).
    #    b. For each swap, calculate the NEW fitness.
    #    c. If NEW fitness < CURRENT fitness:
    #       - ACCEPT the move (update sequence and fitness).
    #       - Break sweep and start a new one (steepest/first-improvement logic).
    # 3. Return the best Solution found.
    
    N = len(cost_matrix)
    
    # 1. Initialize with greedy_initialization if not provided.
    if initial_solution is None:
        sequence = epsilon_greedy_initialization(N, cost_matrix, p_random=0.1)
        current_fitness = calculate_fitness(sequence, cost_matrix)
        current_solution = Solution(sequence=sequence, fitness=current_fitness)
    else:
        current_solution = initial_solution
    
    iterations = 0
    
    # 2. Loop until no improvement (or max_iterations/budget reached):
    while iterations < max_iterations:
        improved = False
        
        #    a. Perform a 'sweep': Iterate through all possible 2-opt swaps (i, j).
        for i in range(N - 1):
            for j in range(i + 1, N):
                # Generate neighbor using 2-opt move
                new_sequence = two_opt_move(current_solution.sequence, i, j)
                
                #    b. For each swap, calculate the NEW fitness.
                new_fitness = calculate_fitness(new_sequence, cost_matrix)
                
                #    c. If NEW fitness < CURRENT fitness:
                if new_fitness < current_solution.fitness:
                    #       - ACCEPT the move (update sequence and fitness).
                    current_solution = Solution(sequence=new_sequence, fitness=new_fitness)
                    improved = True
                    #       - Break sweep and start a new one (steepest/first-improvement logic).
                    break
            
            if improved:
                break
        
        iterations += 1
        
        # If no improvement found, terminate
        if not improved:
            break
    
    # 3. Return the best Solution found.
    return current_solution

def simulated_annealing_solver(cost_matrix, max_iterations, initial_solution=None):
    """
    Runs the 2-opt Hill Climber with Simulated Annealing.
    """
    # 1. Initialize: current_solution, best_solution, and initial Temperature T.
    # 2. Loop max_iterations:
    #    a. Generate a neighbor (e.g., via 2-opt swap).
    #    b. Calculate delta_fitness = (new_fitness - old_fitness).
    #    c. If delta_fitness < 0 (improvement):
    #       - ACCEPT the neighbor.
    #    d. If delta_fitness >= 0 (non-improvement):
    #       - ACCEPT with probability p = exp(-delta_fitness / T).
    #    e. Update T using the temperature schedule:
    #       T_k = 0.1 * (MAX_ITER - CUR_ITER) / MAX_ITER
    # 3. Update best_solution if current_solution is better.
    # 4. Return the best Solution found.
    
    N = len(cost_matrix)
    
    # 1. Initialize: current_solution, best_solution, and initial Temperature T.
    if initial_solution is None:
        sequence = epsilon_greedy_initialization(N, cost_matrix, p_random=0.1)
        current_fitness = calculate_fitness(sequence, cost_matrix)
        current_solution = Solution(sequence=sequence, fitness=current_fitness)
    else:
        current_solution = initial_solution
    
    best_solution = current_solution
    
    # 2. Loop max_iterations:
    for iteration in range(max_iterations):
        #    a. Generate a neighbor (e.g., via 2-opt swap).
        # Select random i and j for 2-opt move
        i, j = sorted(np.random.choice(N, size=2, replace=False))
        new_sequence = two_opt_move(current_solution.sequence, i, j)
        
        #    b. Calculate delta_fitness = (new_fitness - old_fitness).
        new_fitness = calculate_fitness(new_sequence, cost_matrix)
        delta_fitness = new_fitness - current_solution.fitness
        
        #    c. If delta_fitness < 0 (improvement):
        if delta_fitness < 0:
            #       - ACCEPT the neighbor.
            current_solution = Solution(sequence=new_sequence, fitness=new_fitness)
        #    d. If delta_fitness >= 0 (non-improvement):
        else:
            #       - ACCEPT with probability p = exp(-delta_fitness / T).
            #    e. Update T using the temperature schedule:
            #       T_k = 0.1 * (MAX_ITER - CUR_ITER) / MAX_ITER
            T = 0.1 * (max_iterations - iteration) / max_iterations
            
            if T > 0:
                acceptance_probability = np.exp(-delta_fitness / T)
                if np.random.random() < acceptance_probability:
                    current_solution = Solution(sequence=new_sequence, fitness=new_fitness)
        
        # 3. Update best_solution if current_solution is better.
        if current_solution.fitness < best_solution.fitness:
            best_solution = current_solution
    
    # 4. Return the best Solution found.
    return best_solution


def calculate_fitness(sequence, cost_matrix):
    """
    Calculates the total cost (fitness) of a tour sequence.
    """
    total_cost = 0
    for i in range(len(sequence)):
        from_city = sequence[i]
        to_city = sequence[(i + 1) % len(sequence)]
        total_cost += cost_matrix[from_city][to_city]
    return total_cost
