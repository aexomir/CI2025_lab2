# TSP Optimization: Hill Climbing vs Evolutionary Computation

A comprehensive benchmarking framework comparing two meta-heuristic algorithms for solving the Traveling Salesman Problem (TSP).

## 📊 Executive Summary

This project implements and compares two optimization approaches on 22 TSP problem instances:

- **Hill Climbing (HC)** with Simulated Annealing
- **Evolutionary Computation (EC)** with tournament selection and genetic operators

### Key Results

| Metric                               | Value                   |
| ------------------------------------ | ----------------------- |
| **Total Problems Tested**            | 22 instances            |
| **Average HC Execution Time**        | 0.51 seconds            |
| **Average EC Execution Time**        | 59.40 seconds           |
| **Average Improvement (EC over HC)** | 3.99%                   |
| **Best Improvement Achieved**        | 31.36% (problem_r1_200) |

**Performance Highlights:**

- EC significantly outperforms HC on `r1_*` problem types (average 21.59% improvement)
- EC shows mixed results on `r2_*` problems (some degradation due to problem characteristics)
- HC is ~116x faster but less accurate
- EC's longer runtime trades speed for solution quality

---

## 🏗️ Project Structure

```
CI/lab2/
├── README.md                    # This file
├── RESULTS_FORMAT.md           # Detailed results documentation
├── requirements.txt            # Python dependencies
│
├── lab_problems/               # TSP problem instances (.npy files)
│   ├── problem_g_10.npy       # Geometric problems (10-1000 nodes)
│   ├── problem_r1_10.npy      # Random type 1 problems
│   ├── problem_r2_10.npy      # Random type 2 problems
│   └── test_problem.npy       # Test instance
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── main_runner.py         # Entry point for benchmarking
│   │
│   ├── data_problems/         # Data structures
│   │   ├── __init__.py
│   │   └── solution.py        # Solution and SolutionResults classes
│   │
│   ├── solvers/               # Algorithm implementations
│   │   ├── __init__.py
│   │   ├── hc_solver.py       # Hill Climbing + Simulated Annealing
│   │   └── ec_solver.py       # Evolutionary Computation
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       └── test_solvers.py    # Benchmarking framework
│
├── results/                    # Output directory
│   ├── global_summary.csv     # Aggregated results
│   ├── problem_*_summary.csv  # Per-problem HC vs EC comparison
│   ├── problem_*_ec_results.csv   # Detailed EC configurations
│   └── *.npy                  # Binary result objects
│
└── notebooks/                  # Jupyter notebooks for analysis
    ├── 01_analysis.ipynb
    └── 02_custom_test.ipynb
```

---

## 🧮 Algorithms

### 1. Hill Climbing (HC) with Simulated Annealing

**Algorithm Type:** Local Search with Metropolis Criterion

**Key Components:**

- **Initialization:** Epsilon-greedy nearest neighbor (10% random, 90% greedy)
- **Neighborhood:** 2-opt swaps (reverses tour segments)
- **Acceptance:** Simulated Annealing schedule
  - Temperature: `T = 0.1 * (MAX_ITER - CUR_ITER) / MAX_ITER`
  - Accept worse solutions with probability: `exp(-Δfitness / T)`
- **Iterations:** 10,000 fixed

**Characteristics:**

- ✅ Fast execution (~0.5s average)
- ✅ Simple, easy to implement
- ✅ Guaranteed local optima
- ⚠️ May get trapped in local minima
- ⚠️ Performance depends on initial solution

**Implementation:** `src/solvers/hc_solver.py`

- `epsilon_greedy_initialization()` - Creates starting tour
- `two_opt_move()` - Generates neighbors
- `simulated_annealing_solver()` - Main algorithm

---

### 2. Evolutionary Computation (EC)

**Algorithm Type:** Population-based Evolutionary Strategy

**Key Components:**

- **Population:** Mixed initialization (70% greedy, 30% random)
- **Selection:** Tournament selection (τ = 10% of population)
- **Crossover:** Order Crossover (OX) - preserves tour validity
- **Mutation:** Swap mutation (probability: 0.1-0.3)
- **Survival:** Two strategies tested
  - `(μ+λ)` - Elitist: best from parents + offspring
  - `(μ,λ)` - Non-elitist: best from offspring only

**Parameters Tested:**

- Population sizes: 50, 100, 150
- Offspring sizes: 100, 200, 300
- Mutation rates: 0.1, 0.2, 0.3
- Generations: 100
- **Total configurations:** 54 (3×3×3×2)

**Characteristics:**

- ✅ Explores solution space globally
- ✅ Less prone to local optima
- ✅ Better solution quality on average
- ⚠️ Computationally expensive (~60s average)
- ⚠️ Requires parameter tuning

**Implementation:** `src/solvers/ec_solver.py`

- `tournament_selection()` - Parent selection
- `order_crossover_ox()` - Recombination
- `swap_mutation()` - Variation
- `ec_solver()` - Main evolutionary loop

---

## 📈 Results Analysis

### Performance by Problem Type

#### **Geometric Problems (`g_*`)**

Average improvement: **4.76%**

| Problem | Nodes | HC Fitness | EC Fitness | Improvement | EC Time (s) |
| ------- | ----- | ---------- | ---------- | ----------- | ----------- |
| g_10    | 10    | 1497.66    | 1497.66    | 0.00%       | 0.37        |
| g_20    | 20    | 1809.41    | 1755.51    | 2.98%       | 0.43        |
| g_50    | 50    | 2867.87    | 2723.55    | 5.03%       | 2.52        |
| g_100   | 100   | 4513.63    | 4435.11    | 1.74%       | 1.36        |
| g_200   | 200   | 7992.89    | 7502.23    | 6.14%       | 4.66        |
| g_500   | 500   | 16658.68   | 15984.43   | 4.05%       | 98.69       |
| g_1000  | 1000  | 35085.07   | 30055.45   | **14.34%**  | 344.77      |

**Insight:** EC shows increasing advantage with problem size, achieving 14% improvement on the largest instance.

---

#### **Random Type 1 Problems (`r1_*`)**

Average improvement: **21.59%** 🏆 **BEST PERFORMANCE**

| Problem | Nodes | HC Fitness | EC Fitness | Improvement | EC Time (s) |
| ------- | ----- | ---------- | ---------- | ----------- | ----------- |
| r1_10   | 10    | 214.38     | 184.27     | 14.05%      | 0.36        |
| r1_20   | 20    | 382.81     | 337.29     | 11.89%      | 1.29        |
| r1_50   | 50    | 717.82     | 561.29     | 21.81%      | 2.26        |
| r1_100  | 100   | 978.98     | 802.69     | 18.01%      | 3.22        |
| r1_200  | 200   | 1865.98    | 1280.75    | **31.36%**  | 8.62        |
| r1_500  | 500   | 3694.75    | 2891.18    | 21.75%      | 71.57       |
| r1_1000 | 1000  | 7631.20    | 5320.89    | 30.27%      | 372.21      |

**Insight:** EC excels on r1 problems, suggesting these instances have complex fitness landscapes where population-based search is superior.

---

#### **Random Type 2 Problems (`r2_*`)**

Average improvement: **-14.72%** ⚠️ (EC performs worse)

| Problem | Nodes | HC Fitness | EC Fitness | "Improvement" | EC Time (s) |
| ------- | ----- | ---------- | ---------- | ------------- | ----------- |
| r2_10   | 10    | -340.01    | -411.70    | -21.09%       | 0.85        |
| r2_20   | 20    | -691.08    | -845.41    | -22.33%       | 1.02        |
| r2_50   | 50    | -1799.69   | -2244.77   | -24.73%       | 0.67        |
| r2_100  | 100   | -4233.80   | -4621.46   | -9.16%        | 5.52        |
| r2_200  | 200   | -8319.95   | -9432.65   | -13.37%       | 9.74        |
| r2_500  | 500   | -21618.32  | -23401.71  | -8.25%        | 101.32      |
| r2_1000 | 1000  | -43939.12  | -46608.67  | -6.08%        | 274.57      |

**Insight:** Negative fitness values indicate r2 problems are maximization with sign flip. EC's population diversity may be less effective on these smoother landscapes where HC's local search excels.

---

### Optimal EC Configurations

**Most Common Winners:**

- **Strategy:** `(μ,λ)` won on 16/22 problems (non-elitist performs better)
- **Population:** Mixed (50-150), problem-dependent
- **Offspring:** Larger populations (200-300) preferred for complex problems

**Configuration Frequency:**

```
(μ,λ) with pop=150, offspring=300: 8 wins  (best for large/complex)
(μ,λ) with pop=50,  offspring=200: 4 wins  (good for medium)
(μ+λ) with pop=50,  offspring=100: 3 wins  (fast, small problems)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aexomir/CI2025_lab2

# Install dependencies
pip install -r requirements.txt
```

### Running Benchmarks

```bash
# Full benchmark (54 EC configurations per problem)
python src/main_runner.py

# Quick test (16 EC configurations)
python -c "
from src.utils.test_solvers import run_benchmarks
run_benchmarks(ec_config='easy')
"
```

### Analyzing Results

```python
import pandas as pd

# Load global summary
df = pd.read_csv('results/global_summary.csv')

# View top improvements
top5 = df.nlargest(5, 'ec_improvement')
print(top5[['problem', 'hc_fitness', 'ec_best_fitness', 'ec_improvement']])

# Analyze EC configurations for a problem
ec_details = pd.read_csv('results/problem_r1_200_ec_results.csv')
best = ec_details.nsmallest(5, 'best_fitness')
print(best[['best_fitness', 'execution_time', 'strategy', 'population_size']])
```

---

## 📁 Output Files

### Per-Problem Files

- `{problem}_hc_results.npy` - Binary HC results
- `{problem}_ec_best_results.npy` - Binary best EC results
- `{problem}_ec_all_results.npy` - All 54 EC configurations
- `{problem}_summary.csv` - HC vs EC comparison
- `{problem}_ec_results.csv` - Detailed EC configurations

### Global Summary

- `global_summary.csv` - Aggregated results across all problems

See `RESULTS_FORMAT.md` for detailed file format documentation.

---

## 🎯 Key Takeaways

1. **EC vs HC Trade-off:**

   - EC provides 4% average improvement at 116x time cost
   - ROI depends on problem size and quality requirements

2. **Problem-Specific Performance:**

   - EC strongly recommended for `r1_*` type problems (21% improvement)
   - HC sufficient for `r2_*` problems (faster, competitive results)
   - EC scales better with problem size

3. **Parameter Selection:**

   - `(μ,λ)` strategy generally outperforms `(μ+λ)`
   - Larger offspring populations (200-300) beneficial for complex instances
   - Mutation rate 0.1-0.2 works well across problems

4. **Practical Recommendations:**
   - Use HC for: Real-time applications, small problems (<100 nodes), r2-type landscapes
   - Use EC for: High-quality solutions needed, r1-type problems, large instances (>500 nodes)

---

## 🔬 Technical Details

### TSP Representation

- **Encoding:** Permutation of city indices [0, 1, ..., N-1]
- **Fitness:** Total tour length (Euclidean distance sum + return to start)
- **Objective:** Minimization

### Computational Environment

- **Language:** Python 3.8+
- **Parallelization:** joblib (all CPU cores for EC configurations)
- **Random Seed:** 42 (for reproducibility)

### Benchmarking Methodology

1. Load TSP instance from `.npy` file
2. Run HC solver (single run, 10,000 iterations)
3. Run 54 EC configurations in parallel
4. Record best fitness and execution time
5. Save results in multiple formats (binary + CSV)

**Total Runtime (22 problems):** ~22 minutes (HC: 11s, EC: 1307s)

## Acknowledgements / Ethical Note

This project drew comparative HC vs EC benchmarking idea from https://github.com/DjangoRepoMngr/CI2025_lab2;

AI assistants were used to brainstorm structure and minor refinements, but all algorithms, parameter choices, and source code were authored independently without copying—this attribution clarifies influence and ethical use of external and AI resources.
