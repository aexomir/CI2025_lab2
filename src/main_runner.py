from utils.test_solvers import run_benchmarks
import os

if __name__ == "__main__":
    # --- Project Setup ---
    os.makedirs('lab_problems', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Starting TSP Solver Benchmarks...")
    
    # Execute the testing framework
    try:
        run_benchmarks(es_config='default')
    except Exception as e:
        print(f"[ERR] An error occurred during benchmarking: {e}")
    
    print("Main execution finished. Check 'results/' for output files.")