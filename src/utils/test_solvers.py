# src/utils/test_solvers.py

import numpy as np
import os
import sys
import time
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joblib import Parallel, delayed
from typing import List, Dict, Any

# Assuming imports from other src files
from data_problems.solution import SolutionResults, Solution
from solvers.hc_solver import hill_climber_solver, simulated_annealing_solver
from solvers.ec_solver import ec_solver

# --- 1. Parameter Definitions ---

def easy_combinations() -> List[Dict[str, Any]]:
    """Returns a simplified list of EC parameter dictionaries."""
    # Example structure based on README: 16 configurations (2*2*2*2)
    # [
    #   {'mutation_rate': 0.1, 'population_size': 50, 'offspring_size': 100, 'strategy': 'mu+lambda'},
    #   ...
    # ]
    mutation_rates = [0.1, 0.2]
    population_sizes = [50, 100]
    offspring_sizes = [100, 200]
    strategies = ['mu+lambda', 'mu,lambda']
    
    combinations = []
    for mutation_rate in mutation_rates:
        for population_size in population_sizes:
            for offspring_size in offspring_sizes:
                for strategy in strategies:
                    combinations.append({
                        'mutation_rate': mutation_rate,
                        'population_size': population_size,
                        'offspring_size': offspring_size,
                        'strategy': strategy,
                        'generations': 100
                    })
    
    return combinations

def default_combinations() -> List[Dict[str, Any]]:
    """Returns the full list of EC parameter dictionaries for benchmarking."""
    # Example structure based on README: 54 configurations (3*3*3*2)
    mutation_rates = [0.1, 0.2, 0.3]
    population_sizes = [50, 100, 150]
    offspring_sizes = [100, 200, 300]
    strategies = ['mu+lambda', 'mu,lambda']
    
    combinations = []
    for mutation_rate in mutation_rates:
        for population_size in population_sizes:
            for offspring_size in offspring_sizes:
                for strategy in strategies:
                    combinations.append({
                        'mutation_rate': mutation_rate,
                        'population_size': population_size,
                        'offspring_size': offspring_size,
                        'strategy': strategy,
                        'generations': 100
                    })
    
    return combinations

# --- 2. Core Task Runners ---

def _load_problem(file_path: str) -> np.ndarray:
    """Loads a TSP cost matrix from an .npy file."""
    # Use numpy.load(file_path)
    return np.load(file_path)

def _hc_task(problem_name: str, cost_matrix: np.ndarray) -> SolutionResults:
    """
    Runs a single Hill Climber (HC) solver with fixed parameters.
    """
    print(f"Running HC for {problem_name}...")
    # 1. Define fixed HC params (e.g., 10,000 iterations, greedy init, SA enabled)
    # 2. hc_solution = simulated_annealing_solver(cost_matrix, fixed_params)
    # 3. Create SolutionResults object:
    #    - Compress history using run_length_encode(hc_solution.history)
    # 4. Return SolutionResults
    
    # 1. Define fixed HC params (e.g., 10,000 iterations, greedy init, SA enabled)
    max_iterations = 10000
    
    # Track execution time
    start_time = time.time()
    
    # 2. hc_solution = simulated_annealing_solver (SA enabled)
    hc_solution = simulated_annealing_solver(cost_matrix, max_iterations, initial_solution=None)
    
    execution_time = time.time() - start_time
    
    # 3. Create SolutionResults object:
    params = {
        'solver': 'hill_climber_sa',
        'max_iterations': max_iterations
    }
    
    # Since we don't have history tracking yet, use empty history
    history = {'values': [], 'counts': []}
    
    result = SolutionResults(
        best_solution=hc_solution,
        best_fitness=hc_solution.fitness,
        best_sequence=hc_solution.sequence,
        history=history,
        params=params,
        execution_time=execution_time
    )
    
    print(f"HC completed for {problem_name} in {execution_time:.2f}s with fitness {hc_solution.fitness}")
    
    # 4. Return SolutionResults
    return result

def _ec_task(problem_name: str, cost_matrix: np.ndarray, params: Dict[str, Any]) -> SolutionResults:
    """
    Runs a single Evolutionary Computation (EC) configuration.
    """
    # 1. ec_solution = ec_solver(cost_matrix, **params)
    # 2. Create SolutionResults object:
    #    - Compress history
    # 3. Return SolutionResults
    
    # Track execution time
    start_time = time.time()
    
    # 1. ec_solution = ec_solver(cost_matrix, **params)
    ec_solution = ec_solver(
        cost_matrix=cost_matrix,
        population_size=params['population_size'],
        offspring_size=params['offspring_size'],
        generations=params['generations'],
        strategy=params['strategy']
    )
    
    execution_time = time.time() - start_time
    
    # 2. Create SolutionResults object:
    # Since we don't have history tracking yet, use empty history
    history = {'values': [], 'counts': []}
    
    result = SolutionResults(
        best_solution=ec_solution,
        best_fitness=ec_solution.fitness,
        best_sequence=ec_solution.sequence,
        history=history,
        params=params,
        execution_time=execution_time
    )
    
    # 3. Return SolutionResults
    return result

# --- 3. Main Testing Framework ---

def run_benchmarks(problem_dir: str = 'lab_problems/', output_dir: str = 'results/', ec_config='default') -> None:
    """
    Main function to execute the full benchmarking process in parallel.
    """
    # Set reproducibility seed
    np.random.seed(42)
    
    # Get parameter grid
    if ec_config == 'default':
        ec_params_grid = default_combinations()
    elif ec_config == 'easy':
        ec_params_grid = easy_combinations()
    else:
        raise ValueError("Invalid EC configuration name.")

    # Store all results for global summary
    all_summaries = []

    # Iterate through all problem files
    for problem_file in os.listdir(problem_dir):
        if not problem_file.endswith('.npy'):
            continue
            
        problem_name = problem_file.replace('.npy', '')
        file_path = os.path.join(problem_dir, problem_file)
        cost_matrix = _load_problem(file_path)

        # 1. Run Hill Climber (single task)
        hc_result = _hc_task(problem_name, cost_matrix)

        # 2. Run EC Tests in Parallel
        print(f"Starting parallel EC runs for {problem_name} ({len(ec_params_grid)} configs)...")
        
        # joblib.Parallel usage based on README
        ec_results = Parallel(n_jobs=-1)(
            delayed(_ec_task)(problem_name, cost_matrix, params)
            for params in ec_params_grid
        )
        
        # 3. Process and Save EC Results (e.g., aggregate and find best)
        ec_best_result = _process_and_save_ec_results(ec_results, problem_name, output_dir)
        
        # Also save HC result
        _save_results(hc_result, problem_name, 'hc', output_dir)
        
        # Save comparison summary
        _save_summary_to_csv(hc_result, ec_best_result, problem_name, output_dir)
        
        # Collect for global summary
        all_summaries.append({
            'problem': problem_name,
            'hc_fitness': hc_result.best_fitness,
            'hc_time': hc_result.execution_time,
            'ec_best_fitness': ec_best_result.best_fitness,
            'ec_best_time': ec_best_result.execution_time,
            'ec_best_strategy': ec_best_result.params.get('strategy', 'N/A'),
            'ec_best_population': ec_best_result.params.get('population_size', 'N/A'),
            'ec_best_offspring': ec_best_result.params.get('offspring_size', 'N/A')
        })
    
    # Save global summary
    _save_global_summary(all_summaries, output_dir)
    
    print("Benchmarking complete.")


# --- 4. Result Processing/Saving (Helper function) ---

def _save_results(results_obj: SolutionResults, problem_name: str, solver_type: str, output_dir: str):
    """Saves a single SolutionResults object using numpy's format (.npy)."""
    # Naming pattern: problem_{name}_{solver}_results.npy
    # Use numpy.save()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Naming pattern: problem_{name}_{solver}_results.npy
    filename = f"{problem_name}_{solver_type}_results.npy"
    filepath = os.path.join(output_dir, filename)
    
    # Use numpy.save()
    np.save(filepath, results_obj)
    print(f"Saved results to {filepath}")

def _process_and_save_ec_results(ec_results: List[SolutionResults], problem_name: str, output_dir: str) -> SolutionResults:
    """Aggregates all EC runs to find the overall best solution for the problem."""
    # 1. Find the result object with the minimum best_fitness.
    # 2. Save the aggregated best solution data (e.g., to a CSV or JSON).
    
    # 1. Find the result object with the minimum best_fitness.
    best_result = min(ec_results, key=lambda r: r.best_fitness)
    
    # 2. Save the aggregated best solution data
    _save_results(best_result, problem_name, 'ec_best', output_dir)
    
    # Also save all EC results
    all_results_file = os.path.join(output_dir, f"{problem_name}_ec_all_results.npy")
    np.save(all_results_file, ec_results)
    print(f"Saved all {len(ec_results)} EC results to {all_results_file}")
    print(f"Best EC fitness for {problem_name}: {best_result.best_fitness} (time: {best_result.execution_time:.2f}s)")
    
    # Save detailed results to CSV
    _save_ec_results_to_csv(ec_results, problem_name, output_dir)
    
    return best_result


def _save_ec_results_to_csv(ec_results: List[SolutionResults], problem_name: str, output_dir: str):
    """Saves all EC results to a CSV file with fitness and execution time."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV
    results_data = []
    for result in ec_results:
        row = {
            'problem': problem_name,
            'solver': 'EC',
            'best_fitness': result.best_fitness,
            'execution_time': result.execution_time,
            'population_size': result.params.get('population_size', 'N/A'),
            'offspring_size': result.params.get('offspring_size', 'N/A'),
            'generations': result.params.get('generations', 'N/A'),
            'strategy': result.params.get('strategy', 'N/A'),
            'mutation_rate': result.params.get('mutation_rate', 'N/A')
        }
        results_data.append(row)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, f"{problem_name}_ec_results.csv")
    df = pd.DataFrame(results_data)
    df.to_csv(csv_file, index=False)
    print(f"Saved EC results to CSV: {csv_file}")


def _save_summary_to_csv(hc_result: SolutionResults, ec_best_result: SolutionResults, 
                         problem_name: str, output_dir: str):
    """Saves a summary CSV comparing HC and EC best results for a problem."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = [
        {
            'problem': problem_name,
            'solver': 'HC (Simulated Annealing)',
            'best_fitness': hc_result.best_fitness,
            'execution_time': hc_result.execution_time,
            'max_iterations': hc_result.params.get('max_iterations', 'N/A')
        },
        {
            'problem': problem_name,
            'solver': 'EC (Best)',
            'best_fitness': ec_best_result.best_fitness,
            'execution_time': ec_best_result.execution_time,
            'population_size': ec_best_result.params.get('population_size', 'N/A'),
            'offspring_size': ec_best_result.params.get('offspring_size', 'N/A'),
            'generations': ec_best_result.params.get('generations', 'N/A'),
            'strategy': ec_best_result.params.get('strategy', 'N/A')
        }
    ]
    
    summary_file = os.path.join(output_dir, f"{problem_name}_summary.csv")
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False)
    print(f"Saved summary to CSV: {summary_file}")


def _save_global_summary(all_summaries: List[Dict[str, Any]], output_dir: str):
    """Saves a global summary CSV with all problems and their best results."""
    os.makedirs(output_dir, exist_ok=True)
    
    global_file = os.path.join(output_dir, "global_summary.csv")
    df = pd.DataFrame(all_summaries)
    
    # Sort by problem name for easier reading
    df = df.sort_values('problem')
    
    # Add improvement column
    df['ec_improvement'] = ((df['hc_fitness'] - df['ec_best_fitness']) / df['hc_fitness'] * 100).round(2)
    
    df.to_csv(global_file, index=False)
    print(f"\n{'='*60}")
    print(f"Saved global summary to: {global_file}")
    print(f"{'='*60}")
    print("\nSummary Statistics:")
    print(f"Total problems solved: {len(all_summaries)}")
    print(f"Average HC time: {df['hc_time'].mean():.2f}s")
    print(f"Average EC time: {df['ec_best_time'].mean():.2f}s")
    print(f"Average improvement: {df['ec_improvement'].mean():.2f}%")
    print(f"{'='*60}\n")

