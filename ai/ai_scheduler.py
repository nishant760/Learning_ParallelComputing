# ai_scheduler.py
# ------------------------------------------------------------------
# Project: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation
# Description: Smart AI scheduler that uses a trained Random Forest model to
#              evaluate candidate configurations, recommend optimal settings,
#              trigger live C OpenMP execution using warm-up + median methodology,
#              and compare Predicted Performance vs Measured Performance.
# ------------------------------------------------------------------

import os
import sys
import subprocess
import joblib
import pandas as pd
import numpy as np

WARMUP_RUNS = 1
MEASURED_RUNS = 5

def run_c_executable(cmd_list):
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        out = res.stdout
        time_val = None
        pi_val = None
        for line in out.splitlines():
            if "Execution Time (s)" in line:
                time_val = float(line.split(":")[-1].strip())
            elif "Calculated Pi" in line:
                pi_val = float(line.split(":")[-1].strip())
        return time_val, pi_val
    except Exception as e:
        print(f"Execution Error running {cmd_list}: {e}")
        return None, None

def measure_c_median_time(cmd_list):
    # 1. Warm-up run
    for _ in range(WARMUP_RUNS):
        run_c_executable(cmd_list)

    # 2. Measured runs
    times = []
    pi_val = None
    for _ in range(MEASURED_RUNS):
        t, p = run_c_executable(cmd_list)
        if t is not None:
            times.append(t)
            pi_val = p

    if not times:
        return None, None, None

    median_t = float(np.median(times))
    mean_t = float(np.mean(times))
    stddev_t = float(np.std(times))
    return median_t, mean_t, pi_val

def schedule_workload(target_N=80000000):
    model_path = os.path.join("model", "speedup_model.pkl")
    seq_bin = os.path.join("bin", "sequential_pi")
    par_bin = os.path.join("bin", "parallel_pi")

    if not os.path.exists(model_path):
        print("[ERROR] Model file model/speedup_model.pkl missing. Run 'make train' first.")
        sys.exit(1)

    if not os.path.exists(seq_bin) or not os.path.exists(par_bin):
        print("[INFO] C executables missing. Running 'make all'...")
        subprocess.run(["make", "all"], check=True)

    model = joblib.load(model_path)

    # Phase 4 & 6: Candidate configurations (threads = [2, 4, 8])
    schedules = ['static', 'dynamic', 'guided']
    threads_list = [2, 4, 8]
    chunks = [100, 1000, 10000]

    candidates = []
    for sched in schedules:
        for t in threads_list:
            for c in chunks:
                candidates.append({
                    'N': target_N,
                    'threads': t,
                    'schedule': sched,
                    'chunk': c
                })

    candidates_df = pd.DataFrame(candidates)
    predicted_speedups = model.predict(candidates_df)
    candidates_df['predicted_speedup'] = predicted_speedups

    # Rank candidates by predicted speedup
    sorted_candidates = candidates_df.sort_values(by='predicted_speedup', ascending=False).reset_index(drop=True)
    top_config = sorted_candidates.iloc[0]

    print("\n=========================================================================")
    print(f" AI-ASSISTED OPENMP SCHEDULER (Target Workload N = {target_N:,})")
    print("=========================================================================")
    print("\nTop 5 Ranked AI Candidates (by Predicted Speedup):")
    print(sorted_candidates.head(5)[['threads', 'schedule', 'chunk', 'predicted_speedup']].to_string(index=False))

    print("\n-------------------------------------------------------------------------")
    print(" AI OPTIMAL RECOMMENDATION:")
    print(f"   Threads            : {int(top_config['threads'])}")
    print(f"   Scheduling Strategy: {top_config['schedule'].upper()}")
    print(f"   Chunk Size         : {int(top_config['chunk'])}")
    print(f"   Predicted Performance (Speedup) : {top_config['predicted_speedup']:.4f}x")
    print("-------------------------------------------------------------------------")

    print(f"\n[LIVE EXECUTION] Running live C binaries ({WARMUP_RUNS} warm-up + {MEASURED_RUNS} measured runs)...")

    # 1. Measure Sequential Baseline Median Time
    seq_cmd = [seq_bin, str(target_N)]
    seq_median_time, _, seq_pi = measure_c_median_time(seq_cmd)

    # 2. Measure Parallel Recommended Config Median Time
    par_cmd = [
        par_bin,
        str(target_N),
        str(int(top_config['threads'])),
        top_config['schedule'],
        str(int(top_config['chunk']))
    ]
    par_median_time, _, par_pi = measure_c_median_time(par_cmd)

    if seq_median_time is None or par_median_time is None:
        print("[ERROR] Failed to measure execution time.")
        sys.exit(1)

    measured_actual_speedup = seq_median_time / par_median_time
    measured_actual_efficiency = (measured_actual_speedup / top_config['threads']) * 100.0
    pred_speedup = top_config['predicted_speedup']
    abs_error = abs(pred_speedup - measured_actual_speedup)
    pct_error = (abs_error / measured_actual_speedup) * 100.0
    
    exact_pi = 3.14159265358979323846
    pi_err = abs(par_pi - exact_pi)

    print("\n=========================================================================")
    print(" PREDICTED PERFORMANCE vs MEASURED PERFORMANCE")
    print("=========================================================================")
    print(f" Calculated Pi (Parallel)     : {par_pi:.15f}")
    print(f" Negligible Floating-Pt Error : {pi_err:.15e}")
    print(f" Sequential Median Time (T1)  : {seq_median_time:.6f} s")
    print(f" Parallel Median Time (Tp)    : {par_median_time:.6f} s")
    print(f" Predicted Performance        : {pred_speedup:.4f}x Speedup")
    print(f" Measured Performance         : {measured_actual_speedup:.4f}x Speedup")
    print(f" Measured Parallel Efficiency : {measured_actual_efficiency:.2f}%")
    print(f" Prediction Absolute Error    : {abs_error:.4f}")
    print(f" Prediction Percentage Error  : {pct_error:.2f}%")
    print("=========================================================================\n")

    return {
        'N': target_N,
        'recommended_threads': int(top_config['threads']),
        'recommended_schedule': top_config['schedule'],
        'recommended_chunk': int(top_config['chunk']),
        'seq_time': seq_median_time,
        'par_time': par_median_time,
        'actual_speedup': measured_actual_speedup,
        'predicted_speedup': pred_speedup,
        'error': abs_error,
        'pct_error': pct_error
    }

if __name__ == '__main__':
    target = 80000000
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except ValueError:
            pass
    schedule_workload(target)
