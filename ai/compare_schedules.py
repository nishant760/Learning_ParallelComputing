# compare_schedules.py
# ------------------------------------------------------------------
# Project: AI-Assisted Parallel Numerical Computation & Scheduling
# Description: Evaluates and compares all OpenMP scheduling strategies
#              (Static, Dynamic, Guided) live for a given workload.
# ------------------------------------------------------------------

import os
import sys
import subprocess
import pandas as pd

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

def compare_all_schedules(N=100000000, threads=8):
    seq_bin = os.path.join("bin", "sequential_pi")
    par_bin = os.path.join("bin", "parallel_pi")

    if not os.path.exists(seq_bin) or not os.path.exists(par_bin):
        print("[INFO] Compiling C executables...")
        subprocess.run(["make", "all"], check=True)

    print("\n=========================================================================================")
    print(f" COMPARING ALL OPENMP SCHEDULING STRATEGIES (N = {N:,} steps, Threads = {threads})")
    print("=========================================================================================")

    # 1. Run Baseline Sequential
    print("\n[1/2] Running Sequential Baseline...")
    seq_time, seq_pi = run_c_executable([seq_bin, str(N)])
    print(f"      Sequential Time: {seq_time:.6f} s | Calculated Pi: {seq_pi:.15f}")

    # 2. Test All Schedule & Chunk Combinations
    print("\n[2/2] Running Parallel Configurations across Static, Dynamic, and Guided Schedules...")
    schedules = ["static", "dynamic", "guided"]
    chunks = [100, 1000, 10000]

    results = []

    for sched in schedules:
        for chunk in chunks:
            # Min of 3 runs for stability
            best_time = 1e9
            for _ in range(3):
                t_par, _ = run_c_executable([par_bin, str(N), str(threads), sched, str(chunk)])
                if t_par and t_par < best_time:
                    best_time = t_par
            
            speedup = seq_time / best_time
            efficiency = (speedup / threads) * 100.0
            
            results.append({
                'Schedule': sched.upper(),
                'Chunk Size': chunk,
                'Execution Time (s)': best_time,
                'Speedup': speedup,
                'Efficiency (%)': efficiency
            })

    res_df = pd.DataFrame(results)
    sorted_df = res_df.sort_values(by='Execution Time (s)', ascending=True).reset_index(drop=True)

    print("\n-----------------------------------------------------------------------------------------")
    print(" ALL SCHEDULING ALGORITHMS RANKED BY PERFORMANCE (FASTEST TO SLOWEST):")
    print("-----------------------------------------------------------------------------------------")
    for idx, row in sorted_df.iterrows():
        rank = idx + 1
        tag = "🏆 BEST" if rank == 1 else ("⚠️ SLOWEST" if rank == len(sorted_df) else f"   #{rank}")
        print(f"{tag:<10} | Schedule: {row['Schedule']:<7} | Chunk: {row['Chunk Size']:<5} | "
              f"Time: {row['Execution Time (s)']:8.6f} s | Speedup: {row['Speedup']:5.2f}x | Eff: {row['Efficiency (%)']:6.2f}%")
    print("-----------------------------------------------------------------------------------------")

    winner = sorted_df.iloc[0]
    slowest = sorted_df.iloc[-1]

    print("\n=========================================================================================")
    print(" 💡 ANALYSIS & VERDICT")
    print("=========================================================================================")
    print(f" 🏆 WINNER / BEST SCHEDULE: {winner['Schedule']} (Chunk Size: {winner['Chunk Size']})")
    print(f"    - Execution Time : {winner['Execution Time (s)']:.6f} seconds")
    print(f"    - Speedup        : {winner['Speedup']:.2f}x faster than Sequential")
    print(f"    - Efficiency     : {winner['Efficiency (%)']:.2f}% core utilization")
    print("\n 🔍 WHY IS THIS THE BEST?")
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

    print(f"\n ⚠️ WORST CONFIGURATION : {slowest['Schedule']} (Chunk Size: {slowest['Chunk Size']})")
    print(f"    - Execution Time : {slowest['Execution Time (s)']:.6f} seconds ({slowest['Execution Time (s)'] / winner['Execution Time (s)']:.2f}x slower than winner!)")
    if slowest['Schedule'] == 'DYNAMIC' and slowest['Chunk Size'] == 100:
        print("    - EXPLANATION: Small chunk size (100) with DYNAMIC scheduling forces 1,000,000 queue lock")
        print("      requests. Mutex synchronization locking overhead completely dominates calculation time!")
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
