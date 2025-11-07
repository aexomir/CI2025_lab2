# src/solvers/ec_solver.py

# Assuming data_models and utils are available and __init__.py files are in place.
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_problems.solution import Solution
from solvers.hc_solver import epsilon_greedy_initialization
from typing import List, Dict, Any, Union

# ====================== Genetic Operators ======================

def tournament_selection(population, tau: int):
    """
    Selects a parent using tournament selection of size tau (τ).
    Returns the Solution object with the best (lowest) fitness from the tau competitors.
    """
    # Pseudocode:
    # 1. Randomly select 'tau' individuals from the population.
    # 2. The individual with the BEST FITNESS (minimum cost) among the 'tau' is the winner.
    # 3. Return the selected parent.
    tournament = np.random.choice(population, size=tau, replace=False)
    winner = min(tournament, key=lambda sol: sol.fitness)
    return winner


def order_crossover_ox(parent1_sequence: List[int], parent2_sequence: List[int]) -> List[int]:
    """
    Performs Order Crossover (OX) to generate a feasible TSP offspring sequence.
    """
    # Pseudocode based on README:
    # 1. Select two random cut points (slice) in parent1.
    # 2. Copy the segment between the cuts from parent1 to the offspring.
    # 3. Fill the remaining empty positions in the offspring:
    #    - Iterate through parent2's sequence starting from the second cut point.
    #    - Insert cities from parent2 into the offspring, skipping cities already
    #      copied from parent1, maintaining the ORDER from parent2.
    # 4. Return the new offspring sequence.
    n = len(parent1_sequence)
    
    # 1. Select two random cut points
    cut1, cut2 = sorted(np.random.choice(n, size=2, replace=False))
    
    # 2. Copy the segment between the cuts from parent1 to the offspring
    offspring = [None] * n
    offspring[cut1:cut2] = parent1_sequence[cut1:cut2]
    
    # 3. Fill the remaining empty positions in the offspring
    # Iterate through parent2's sequence starting from the second cut point
    parent2_order = parent2_sequence[cut2:] + parent2_sequence[:cut2]
    
    # Insert cities from parent2 into the offspring, skipping cities already copied from parent1
    offspring_idx = cut2
    for city in parent2_order:
        if city not in offspring:
            offspring[offspring_idx] = city
            offspring_idx = (offspring_idx + 1) % n
    
    # 4. Return the new offspring sequence
    return offspring


def swap_mutation(sequence: List[int], mutation_rate: float) -> List[int]:
    """
    Applies swap mutation with probability 'mutation_rate' to the sequence.
    """
    # Pseudocode:
    # 1. With probability 'mutation_rate' (per offspring):
    #    a. Select two random positions (indices) in the sequence.
    #    b. Swap the elements at those two positions.
    # 2. Return the potentially mutated sequence.
    mutated_sequence = sequence.copy()
    
    # 1. With probability 'mutation_rate' (per offspring):
    if np.random.random() < mutation_rate:
        # a. Select two random positions (indices) in the sequence
        idx1, idx2 = np.random.choice(len(sequence), size=2, replace=False)
        # b. Swap the elements at those two positions
        mutated_sequence[idx1], mutated_sequence[idx2] = mutated_sequence[idx2], mutated_sequence[idx1]
    
    # 2. Return the potentially mutated sequence
    return mutated_sequence

# ====================== EC Solver Core ======================

def ec_solver(
    cost_matrix: np.ndarray, 
    population_size: int, 
    offspring_size: int, 
    generations: int, 
    strategy: str = 'mu+lambda', 
    **kwargs: Any
):
    """
    Runs the Evolutionary Computation (EC) solver, implementing selection 
    and survival strategies typically found in ES/GAs.
    
    strategy: 'mu+lambda' (elitist) or 'mu,lambda' (non-elitist).
    """
    # 1. Initialization:
    #    - Create an initial population (e.g., mixing greedy and random solutions).
    #    - Set parameters: μ, λ, τ (from kwargs), mutation_rate (from kwargs).
    # 2. Loop 'generations':
    #    a. Generate Offspring (λ individuals):
    #       - Loop λ times:
    #         i. Select two parents using tournament_selection (τ).
    #         ii. Generate an offspring using order_crossover_ox.
    #         iii. Mutate the offspring using swap_mutation.
    #         iv. Calculate the offspring's fitness (using cost_matrix).
    #    b. Survivor Selection (Next Population):
    #       - Collect all parents and offspring into a combined pool.
    #       - If strategy == 'mu+lambda':
    #         - Select the BEST μ individuals from (Parents ∪ Offspring).
    #       - If strategy == 'mu,lambda':
    #         - Select the BEST μ individuals from (Offspring ONLY).
    # 3. Update best Solution found across all generations.
    # 4. Return the best Solution and the history of the best fitness per generation.
    
    N = len(cost_matrix)
    mu = population_size
    lambda_ = offspring_size
    tau = kwargs.get('tau', max(2, int(mu * 0.1)))  # Tournament size (τ)
    mutation_rate = kwargs.get('mutation_rate', 0.1)  # Probability of mutation per offspring
    
    # 1. Initialization:
    #    - Create an initial population (e.g., mixing greedy and random solutions).
    population = []
    num_greedy = int(mu * 0.7)
    num_random = mu - num_greedy
    
    # Generate greedy solutions
    for _ in range(num_greedy):
        greedy_sequence = epsilon_greedy_initialization(N, cost_matrix, p_random=0.1)
        fitness = calculate_fitness(greedy_sequence, cost_matrix)
        population.append(Solution(sequence=greedy_sequence, fitness=fitness))
    
    # Generate random solutions
    for _ in range(num_random):
        random_sequence = epsilon_greedy_initialization(N, cost_matrix, p_random=1.0)
        fitness = calculate_fitness(random_sequence, cost_matrix)
        population.append(Solution(sequence=random_sequence, fitness=fitness))
    
    # Track the best solution found across all generations
    best_solution = min(population, key=lambda sol: sol.fitness)
    
    # 2. Loop 'generations':
    for gen in range(generations):
        #    a. Generate Offspring (λ individuals):
        offspring = []
        
        #       - Loop λ times:
        for _ in range(lambda_):
            #         i. Select two parents using tournament_selection (τ).
            parent1 = tournament_selection(population, tau)
            parent2 = tournament_selection(population, tau)
            
            #         ii. Generate an offspring using order_crossover_ox.
            offspring_sequence = order_crossover_ox(parent1.sequence, parent2.sequence)
            
            #         iii. Mutate the offspring using swap_mutation.
            offspring_sequence = swap_mutation(offspring_sequence, mutation_rate)
            
            #         iv. Calculate the offspring's fitness (using cost_matrix).
            offspring_fitness = calculate_fitness(offspring_sequence, cost_matrix)
            offspring.append(Solution(sequence=offspring_sequence, fitness=offspring_fitness))
        
        #    b. Survivor Selection (Next Population):
        #       - Collect all parents and offspring into a combined pool.
        #       - If strategy == 'mu+lambda':
        if strategy == 'mu+lambda':
            #         - Select the BEST μ individuals from (Parents ∪ Offspring).
            combined = population + offspring
            population = sorted(combined, key=lambda sol: sol.fitness)[:mu]
        #       - If strategy == 'mu,lambda':
        elif strategy == 'mu,lambda':
            #         - Select the BEST μ individuals from (Offspring ONLY).
            population = sorted(offspring, key=lambda sol: sol.fitness)[:mu]
        
        # 3. Update best Solution found across all generations.
        current_best = min(population, key=lambda sol: sol.fitness)
        if current_best.fitness < best_solution.fitness:
            best_solution = current_best
    
    # 4. Return the best Solution and the history of the best fitness per generation.
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