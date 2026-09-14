# compare_schedules.py
# ------------------------------------------------------------------
# Project: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation
# Description: Evaluates and compares all OpenMP scheduling strategies
#              (Static, Dynamic, Guided) live using median execution time.
# ------------------------------------------------------------------

import os
import sys
import subprocess
import pandas as pd
import numpy as np

WARMUP_RUNS = 1
MEASURED_RUNS = 5

def run_c_executable(cmd_list):
    try:
        res = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        time_val = None
        pi_val = None
        for line in res.stdout.splitlines():
            if "Execution Time (s)" in line:
                time_val = float(line.split(":")[-1].strip())
            elif "Calculated Pi" in line:
                pi_val = float(line.split(":")[-1].strip())
        return time_val, pi_val
    except Exception as e:
        print(f"Error running {cmd_list}: {e}")
        return None, None

def measure_median_time(cmd_list):
    for _ in range(WARMUP_RUNS):
        run_c_executable(cmd_list)
    
    times = []
    for _ in range(MEASURED_RUNS):
        t, _ = run_c_executable(cmd_list)
        if t is not None:
            times.append(t)
            
    return float(np.median(times)) if times else None

def compare_all_schedules(N=100000000, threads=8):
    seq_bin = os.path.join("bin", "sequential_pi")
    par_bin = os.path.join("bin", "parallel_pi")

    if not os.path.exists(seq_bin) or not os.path.exists(par_bin):
        print("[INFO] Compiling C executables...")
        subprocess.run(["make", "all"], check=True)

    print("\n=========================================================================================")
    print(f" COMPARING ALL OPENMP SCHEDULING STRATEGIES (N = {N:,} steps, Threads = {threads})")
    print(f" (Methodology: {WARMUP_RUNS} Warm-up + {MEASURED_RUNS} Measured Runs -> Median Execution Time)")
    print("=========================================================================================")

    # 1. Run Baseline Sequential
    print("\n[1/2] Running Sequential Baseline...")
    seq_time = measure_median_time([seq_bin, str(N)])
    print(f"      Sequential Median Time: {seq_time:.6f} s")

    # 2. Test All Schedule & Chunk Combinations
    print("\n[2/2] Running Parallel Configurations across Static, Dynamic, and Guided Schedules...")
    schedules = ["static", "dynamic", "guided"]
    chunks = [100, 1000, 10000]

    results = []

    for sched in schedules:
        for chunk in chunks:
            par_time = measure_median_time([par_bin, str(N), str(threads), sched, str(chunk)])
            
            speedup = seq_time / par_time
            efficiency = (speedup / threads) * 100.0
            
            results.append({
                'Schedule': sched.upper(),
                'Chunk Size': chunk,
                'Median Time (s)': par_time,
                'Speedup': speedup,
                'Efficiency (%)': efficiency
            })

    res_df = pd.DataFrame(results)
    sorted_df = res_df.sort_values(by='Median Time (s)', ascending=True).reset_index(drop=True)

    print("\n-----------------------------------------------------------------------------------------")
    print(" ALL SCHEDULING ALGORITHMS RANKED BY PERFORMANCE (FASTEST TO SLOWEST):")
    print("-----------------------------------------------------------------------------------------")
    for idx, row in sorted_df.iterrows():
        rank = idx + 1
        tag = "🏆 BEST" if rank == 1 else ("⚠️ SLOWEST" if rank == len(sorted_df) else f"   #{rank}")
        print(f"{tag:<10} | Schedule: {row['Schedule']:<7} | Chunk: {row['Chunk Size']:<5} | "
              f"Time: {row['Median Time (s)']:8.6f} s | Speedup: {row['Speedup']:5.2f}x | Eff: {row['Efficiency (%)']:6.2f}%")
    print("-----------------------------------------------------------------------------------------")

    winner = sorted_df.iloc[0]
    slowest = sorted_df.iloc[-1]

    print("\n=========================================================================================")
    print(" 💡 ANALYSIS & VERDICT")
    print("=========================================================================================")
    print(f" 🏆 BEST SCHEDULE FOR TESTED SYSTEM: {winner['Schedule']} (Chunk Size: {winner['Chunk Size']})")
    print(f"    - Median Time    : {winner['Median Time (s)']:.6f} seconds")
    print(f"    - Speedup        : {winner['Speedup']:.2f}x faster than Sequential baseline")
    print(f"    - Efficiency     : {winner['Efficiency (%)']:.2f}% core utilization")
    print("\n 🔍 WHY THIS PERFORMANCE DIFFERENCE OCCURRED:")
    if winner['Schedule'] == 'GUIDED':
        print("    - GUIDED scheduling starts with large chunk sizes and exponentially shrinks them.")
        print("    - This minimizes task-queue synchronization locking overhead at loop start while")
        print("      providing dynamic load balancing near loop completion.")
    elif winner['Schedule'] == 'STATIC':
        print("    - STATIC scheduling assigns equal iteration blocks at start with zero queue overhead.")
        print("    - Since midpoint numerical integration has uniform iteration computation, static")
        print("      incurs minimal overhead and achieves optimal CPU pipeline throughput.")
    else:
        print("    - DYNAMIC scheduling dynamically assigned work items as threads became available.")

    print(f"\n ⚠️ SLOWEST CONFIGURATION : {slowest['Schedule']} (Chunk Size: {slowest['Chunk Size']})")
    print(f"    - Median Time    : {slowest['Median Time (s)']:.6f} seconds ({slowest['Median Time (s)'] / winner['Median Time (s)']:.2f}x slower than winner)")
    if slowest['Schedule'] == 'DYNAMIC' and slowest['Chunk Size'] == 100:
        print("    - EXPLANATION: Small chunk size (100) with DYNAMIC scheduling introduces runtime")
        print("      scheduling overhead as threads frequently request additional chunks.")
    print("=========================================================================================\n")

if __name__ == '__main__':
    N_val = 100000000
    t_val = 8
    if len(sys.argv) > 1:
        try: N_val = int(sys.argv[1])
        except ValueError: pass
    if len(sys.argv) > 2:
        try: t_val = int(sys.argv[2])
        except ValueError: pass

    compare_all_schedules(N_val, t_val)
