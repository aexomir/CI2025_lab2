# Results Format Documentation

## Overview

The benchmarking system now tracks execution time and stores results in multiple formats for analysis.

## Data Structure Changes

### SolutionResults Class

Updated to include execution time:

```python
@dataclass
class SolutionResults:
    best_solution: Solution
    best_fitness: Union[float, int]
    best_sequence: List[int]
    history: dict
    params: dict
    execution_time: float  # NEW: Time in seconds
```

## Output Files Generated

For each problem (e.g., `problem_g_10`), the following files are created:

### 1. Individual Problem Results

#### `{problem}_hc_results.npy`

- Binary format with HC solver results
- Contains: best solution, fitness, execution time, parameters

#### `{problem}_ec_best_results.npy`

- Binary format with best EC configuration results
- Contains: best solution, fitness, execution time, parameters

#### `{problem}_ec_all_results.npy`

- Binary format with ALL EC configurations tested
- Array of SolutionResults objects

### 2. CSV Files (Human-Readable)

#### `{problem}_ec_results.csv`

Detailed results for all EC configurations:

- `problem`: Problem name
- `solver`: Always 'EC'
- `best_fitness`: Final fitness value
- `execution_time`: Time in seconds
- `population_size`: Population size used
- `offspring_size`: Offspring size used
- `generations`: Number of generations
- `strategy`: 'mu+lambda' or 'mu,lambda'
- `mutation_rate`: Mutation probability

#### `{problem}_summary.csv`

Comparison of HC vs. Best EC:

- Row 1: HC (Simulated Annealing) results
- Row 2: EC (Best configuration) results
- Columns: problem, solver, best_fitness, execution_time, solver-specific params

### 3. Global Summary

#### `global_summary.csv`

Aggregated results across ALL problems:

- `problem`: Problem name
- `hc_fitness`: HC best fitness
- `hc_time`: HC execution time (seconds)
- `ec_best_fitness`: EC best fitness
- `ec_best_time`: EC execution time (seconds)
- `ec_best_strategy`: Strategy of best EC config
- `ec_best_population`: Population size of best EC config
- `ec_best_offspring`: Offspring size of best EC config
- `ec_improvement`: Percentage improvement of EC over HC

## Console Output

During execution, you'll see:

```
Running HC for problem_g_10...
HC completed for problem_g_10 in 5.23s with fitness 1234.56
Starting parallel EC runs for problem_g_10 (54 configs)...
Best EC fitness for problem_g_10: 1150.32 (time: 12.45s)
```

At the end:

```
============================================================
Saved global summary to: results/global_summary.csv
============================================================

Summary Statistics:
Total problems solved: 21
Average HC time: 5.43s
Average EC time: 11.28s
Average improvement: 6.82%
============================================================
```

## Usage Example

```python
from src.utils.test_solvers import run_benchmarks

# Run with default configuration (54 EC configs)
run_benchmarks(
    problem_dir='lab_problems/',
    output_dir='results/',
    ec_config='default'
)

# Run with easy configuration (16 EC configs) - faster
run_benchmarks(
    problem_dir='lab_problems/',
    output_dir='results/',
    ec_config='easy'
)
```

## Analyzing Results

### Load individual results:

```python
import numpy as np
result = np.load('results/problem_g_10_hc_results.npy', allow_pickle=True).item()
print(f"Fitness: {result.best_fitness}")
print(f"Time: {result.execution_time}s")
```

### Analyze CSV summaries:

```python
import pandas as pd

# View global summary
df = pd.read_csv('results/global_summary.csv')
print(df[['problem', 'hc_fitness', 'ec_best_fitness', 'ec_improvement']])

# View detailed EC results for a problem
ec_details = pd.read_csv('results/problem_g_10_ec_results.csv')
best_configs = ec_details.nsmallest(5, 'best_fitness')
print(best_configs)
```
