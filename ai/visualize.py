# visualize.py
# ------------------------------------------------------------------
# Project: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation
# Description: Generates performance analysis graphs using matplotlib
#              from empirical OpenMP benchmark dataset.
# ------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def generate_plots():
    csv_path = os.path.join("data", "performance.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("data/performance.csv missing. Run 'make run-benchmark' first.")

    df = pd.read_csv(csv_path)
    os.makedirs("results", exist_ok=True)

    # Use median_time column from revised CSV format
    time_col = 'median_time' if 'median_time' in df.columns else 'execution_time'

    # Filter standard static schedule with chunk=1000 for standard thread scaling plots
    std_df = df[(df['schedule'] == 'static') & (df['chunk'] == 1000)].copy()

    input_sizes = sorted(df['N'].unique())
    main_sizes = [s for s in input_sizes if s in [1000000, 10000000, 50000000, 100000000]]
    if not main_sizes:
        main_sizes = input_sizes[:4]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. Execution Time vs Threads
    plt.figure(figsize=(8, 5))
    for idx, N_val in enumerate(main_sizes):
        sub = std_df[std_df['N'] == N_val]
        if not sub.empty:
            label_str = f"N = {N_val // 1000000}M" if N_val >= 1000000 else f"N = {N_val}"
            plt.plot(sub['threads'], sub[time_col], marker='o', linewidth=2.0, color=colors[idx % len(colors)], label=label_str)

    plt.title("Median Execution Time vs. Thread Count", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of OpenMP Threads", fontsize=11)
    plt.ylabel("Median Execution Time (seconds)", fontsize=11)
    plt.xticks([1, 2, 4, 8])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Workload Size", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "execution_time_vs_threads.png"), dpi=300)
    plt.close()
    print("Saved results/execution_time_vs_threads.png")

    # 2. Speedup vs Threads
    plt.figure(figsize=(8, 5))
    threads_ref = np.array([1, 2, 4, 8])
    plt.plot(threads_ref, threads_ref, 'k--', linewidth=1.5, label="Ideal Speedup (Linear)")

    for idx, N_val in enumerate(main_sizes):
        sub = std_df[std_df['N'] == N_val]
        if not sub.empty:
            label_str = f"N = {N_val // 1000000}M" if N_val >= 1000000 else f"N = {N_val}"
            plt.plot(sub['threads'], sub['speedup'], marker='s', linewidth=2.0, color=colors[idx % len(colors)], label=label_str)

    plt.title("Speedup vs. Thread Count (Median Baseline)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of OpenMP Threads", fontsize=11)
    plt.ylabel("Speedup (S = T1_median / Tp_median)", fontsize=11)
    plt.xticks([1, 2, 4, 8])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Workload Size", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "speedup_vs_threads.png"), dpi=300)
    plt.close()
    print("Saved results/speedup_vs_threads.png")

    # 3. Efficiency vs Threads
    plt.figure(figsize=(8, 5))
    plt.axhline(100, color='k', linestyle='--', linewidth=1.5, label="100% Reference Efficiency")

    for idx, N_val in enumerate(main_sizes):
        sub = std_df[std_df['N'] == N_val]
        if not sub.empty:
            label_str = f"N = {N_val // 1000000}M" if N_val >= 1000000 else f"N = {N_val}"
            plt.plot(sub['threads'], sub['efficiency'], marker='^', linewidth=2.0, color=colors[idx % len(colors)], label=label_str)

    plt.title("Parallel Efficiency vs. Thread Count", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of OpenMP Threads", fontsize=11)
    plt.ylabel("Efficiency (%)", fontsize=11)
    plt.xticks([1, 2, 4, 8])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Workload Size", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "efficiency_vs_threads.png"), dpi=300)
    plt.close()
    print("Saved results/efficiency_vs_threads.png")

    # 4. Schedule Comparison (Static vs Dynamic vs Guided for N=100M, T=8)
    plt.figure(figsize=(9, 5))
    large_N = max(main_sizes)
    sched_df = df[(df['N'] == large_N) & (df['threads'] == 8)].copy()

    if not sched_df.empty:
        pivot_df = sched_df.pivot(index='chunk', columns='schedule', values='speedup')
        pivot_df = pivot_df[['static', 'dynamic', 'guided']]
        ax = pivot_df.plot(kind='bar', figsize=(9, 5), width=0.75, color=['#2b5c8f', '#d95f02', '#7570b3'])
        plt.title(f"OpenMP Scheduling Policy Comparison (N = {large_N//1000000}M, 8 Threads)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Chunk Size", fontsize=11)
        plt.ylabel("Speedup", fontsize=11)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend(title="Schedule Strategy", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join("results", "schedule_comparison.png"), dpi=300)
        plt.close()
        print("Saved results/schedule_comparison.png")

    # 5. Workload Scaling
    plt.figure(figsize=(8, 5))
    workload_df = std_df[std_df['threads'] == 8].copy()
    if not workload_df.empty:
        plt.plot(workload_df['N'] / 1e6, workload_df['speedup'], marker='D', color='#2ca02c', linewidth=2.2, markersize=7)
        plt.title("Speedup Scaling Across Problem Sizes (8 Threads)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Input Size N (Millions)", fontsize=11)
        plt.ylabel("Speedup (8 Threads)", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join("results", "workload_scaling.png"), dpi=300)
        plt.close()
        print("Saved results/workload_scaling.png")

    # 6. Predicted vs Actual Speedup (Unseen Workload N=75M)
    model_path = os.path.join("model", "speedup_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        unseen_df = df[df['N'] == 75000000].copy()
        if unseen_df.empty:
            unseen_df = df.sample(n=36, random_state=42).copy()

        preds = model.predict(unseen_df[['N', 'threads', 'schedule', 'chunk']])
        unseen_df['predicted_speedup'] = preds

        plt.figure(figsize=(7, 6))
        plt.scatter(unseen_df['speedup'], unseen_df['predicted_speedup'], color='#d62728', alpha=0.8, edgecolors='k', s=60)
        min_val = min(unseen_df['speedup'].min(), unseen_df['predicted_speedup'].min())
        max_val = max(unseen_df['speedup'].max(), unseen_df['predicted_speedup'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label="1:1 Reference Line")

        plt.title("Predicted vs. Measured Speedup (Unseen N = 75M)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Empirical Measured Speedup", fontsize=11)
        plt.ylabel("Random Forest Predicted Speedup", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join("results", "predicted_vs_actual.png"), dpi=300)
        plt.close()
        print("Saved results/predicted_vs_actual.png")

    print("All 6 visualization plots generated successfully in results/ directory!")

if __name__ == '__main__':
    generate_plots()
