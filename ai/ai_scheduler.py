# ai_scheduler.py
# ------------------------------------------------------------------
# Project: AI-Assisted Parallel Numerical Computation & Scheduling
# Description: Smart AI scheduler that evaluates OpenMP thread and scheduling
#              configurations using a trained Random Forest model.
#              Recommends optimal settings for unseen workloads and compares
#              predicted vs actual speedup via live C execution.
# ------------------------------------------------------------------

import os
import sys
import subprocess
import joblib
import pandas as pd
import numpy as np

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

    # Candidate configurations
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

    # Sort candidates by predicted speedup
    sorted_candidates = candidates_df.sort_values(by='predicted_speedup', ascending=False).reset_index(drop=True)

    top_config = sorted_candidates.iloc[0]

    print("\n=========================================================================")
    print(f" AI-ASSISTED OPENMP SCHEDULER (Target Workload N = {target_N:,})")
    print("=========================================================================")
    print("\nTop 5 AI Predicted Optimal Configurations:")
    print(sorted_candidates.head(5)[['threads', 'schedule', 'chunk', 'predicted_speedup']].to_string(index=False))

    print("\n-------------------------------------------------------------------------")
    print(" AI OPTIMAL RECOMMENDATION:")
    print(f"   Threads            : {int(top_config['threads'])}")
    print(f"   Scheduling Strategy: {top_config['schedule'].upper()}")
    print(f"   Chunk Size         : {int(top_config['chunk'])}")
    print(f"   Predicted Speedup  : {top_config['predicted_speedup']:.4f}x")
    print("-------------------------------------------------------------------------")

    print("\n[LIVE EXECUTION] Triggering compiled C binaries to benchmark prediction...")

    # 1. Run Sequential
    seq_time, seq_pi = run_c_executable([seq_bin, str(target_N)])

    # 2. Run Parallel with Recommended Config
    par_time, par_pi = run_c_executable([
        par_bin,
        str(target_N),
        str(int(top_config['threads'])),
        top_config['schedule'],
        str(int(top_config['chunk']))
    ])

    if seq_time is None or par_time is None:
        print("[ERROR] Failed to measure actual execution time.")
        sys.exit(1)

    actual_speedup = seq_time / par_time
    actual_efficiency = (actual_speedup / top_config['threads']) * 100.0
    pred_speedup = top_config['predicted_speedup']
    abs_error = abs(pred_speedup - actual_speedup)
    pct_error = (abs_error / actual_speedup) * 100.0

    print("\n=========================================================================")
    print(" LIVE EXPERIMENTAL RESULTS vs AI PREDICTION")
    print("=========================================================================")
    print(f" Calculated Pi (Sequential) : {seq_pi:.15f}")
    print(f" Calculated Pi (Parallel)   : {par_pi:.15f}")
    print(f" Sequential Execution Time  : {seq_time:.6f} s")
    print(f" Parallel Execution Time    : {par_time:.6f} s")
    print(f" Measured Actual Speedup    : {actual_speedup:.4f}x")
    print(f" Measured Parallel Efficiency: {actual_efficiency:.2f}%")
    print(f" AI Predicted Speedup       : {pred_speedup:.4f}x")
    print(f" Absolute Prediction Error  : {abs_error:.4f}")
    print(f" Percentage Error           : {pct_error:.2f}%")
    print("=========================================================================\n")

    return {
        'N': target_N,
        'recommended_threads': int(top_config['threads']),
        'recommended_schedule': top_config['schedule'],
        'recommended_chunk': int(top_config['chunk']),
        'seq_time': seq_time,
        'par_time': par_time,
        'actual_speedup': actual_speedup,
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
