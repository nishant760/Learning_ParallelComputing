# Professor Demonstration Guide
## AI-Assisted OpenMP Scheduling for Parallel Numerical Computation

This step-by-step guide outlines the recommended demonstration sequence when presenting the project to a professor or evaluation panel.

---

### Step 1: Compile the C/OpenMP Programs
- **Command**:
  ```bash
  make all
  ```
- **What to Observe**:
  GCC compiles `src/sequential_pi.c`, `src/parallel_pi.c`, and `src/benchmark.c` with `-O3 -fopenmp` optimization into the `bin/` directory with zero compiler warnings.
- **Concept to Explain**:
  Explain the build pipeline. All binaries use `-O3` optimization and OpenMP multi-threading flags enabled consistently so that baseline sequential $T_1$ and parallel $T_p$ executions are compiled under identical optimization rules.

---

### Step 2: Run Sequential Baseline Execution
- **Command**:
  ```bash
  ./bin/sequential_pi 100000000
  ```
- **What to Observe**:
  Single-threaded execution output for $N = 100,000,000$ steps (approx. 0.088 seconds wall-clock time) and the calculated $\pi$ value (`3.141592653589793`).
- **Concept to Explain**:
  Explain midpoint numerical integration: $\pi = \int_0^1 \frac{4}{1+x^2} dx \approx h \sum_{i=0}^{N-1} \frac{4}{1+x_i^2}$. Highlight that $\pi$ calculation serves as the benchmark workload for OpenMP performance tuning.

---

### Step 3: Run OpenMP Parallel Execution
- **Command**:
  ```bash
  ./bin/parallel_pi 100000000 8 guided 10000
  ```
- **What to Observe**:
  Multi-threaded execution on 8 threads running `guided` schedule with chunk size 10,000 (approx. 0.015 seconds wall-clock time).
- **Concept to Explain**:
  Explain OpenMP loop parallelization (`#pragma omp parallel for reduction(+:sum) schedule(runtime)`). Highlight how `#pragma omp reduction(+:sum)` creates thread-private accumulation variables to avoid race conditions on the shared sum.

---

### Step 4: Live Schedule & Chunk Comparison
- **Command**:
  ```bash
  ./venv/bin/python ai/compare_schedules.py 100000000 8
  ```
- **What to Observe**:
  Terminal outputs a ranked performance table comparing `STATIC`, `DYNAMIC`, and `GUIDED` schedules across chunk sizes (100, 1000, 10000) using median execution times over 5 measured runs (after 1 warm-up run).
- **Concept to Explain**:
  - `GUIDED` starts with large chunk sizes and exponentially shrinks them, minimizing task-queue locking overhead at loop start while maintaining load balance near the end.
  - `DYNAMIC` with small chunk size (100) incurs heavy runtime scheduling overhead as threads frequently request additional chunks from the shared queue.

---

### Step 5: Inspect Empirical Dataset
- **Command**:
  ```bash
  head -n 10 data/performance.csv
  ```
- **What to Observe**:
  The CSV header `N,threads,schedule,chunk,median_time,mean_time,stddev_time,min_time,speedup,efficiency,pi_error` and real empirical rows generated on the host system.
- **Concept to Explain**:
  Explain the empirical benchmark methodology: 1 warm-up run + 5 measured runs per configuration. Primary speedup ($S = T_1 / T_p$) and efficiency ($E = S / P \times 100\%$) are calculated using **median execution times** to eliminate measurement noise.

---

### Step 6: Train Random Forest Machine Learning Model
- **Command**:
  ```bash
  make train
  ```
- **What to Observe**:
  Random Forest Regressor training logs, GroupKFold cross-validation metrics (grouped by workload size $N$), and zero-shot holdout validation on unseen workload $N = 75,000,000$ ($R^2 \approx 0.9992$, MAPE $< 1\%$).
- **Concept to Explain**:
  - Feature matrix: `N`, `threads` (2, 4, 8), `schedule` (one-hot), `chunk`.
  - Target: `speedup`.
  - Explain why regression error metrics ($R^2$, MAE, RMSE, MAPE) are reported instead of classification accuracy.

---

### Step 7: Run AI-Assisted Scheduler for Unseen Workload
- **Command**:
  ```bash
  ./venv/bin/python ai/ai_scheduler.py 80000000
  ```
- **What to Observe**:
  The AI scheduler evaluates 27 candidate configurations for unseen $N = 80\text{M}$ steps, ranks them by predicted speedup, outputs the top recommendation, executes compiled C binaries live, and displays:
  - **Predicted Performance**: e.g., `5.78x Speedup`
  - **Measured Performance**: e.g., `4.11x Speedup`
  - **Prediction Error**: e.g., MAE / Percentage Error.
- **Concept to Explain**:
  Clarify that the Random Forest model recommends configuration parameters based on learned performance patterns, while OpenMP and the OS execute the physical workload.

---

### Step 8: Open Interactive Streamlit Academic Dashboard
- **Command**:
  ```bash
  make app
  ```
- **What to Observe**:
  Open web browser at `http://localhost:8501`. Walk through the 4 logical tabs:
  1. **Project Overview**: Problem statement, offline training & online recommendation architecture diagrams, system specs (CPU cores, Python version), hardware dependence disclaimer.
  2. **Live Experiment**: Single run configuration parameters ($N=50\text{M}$ Quick Demo), live metrics display, numerical precision error check, and AI prediction.
  3. **Schedule Comparison**: Interactive live comparison tool for Static vs Dynamic vs Guided schedules.
  4. **AI Recommendation**: Full candidate ranking table and saved model visualizations.
- **Concept to Explain**:
  Summarize the central project story: *"Measure OpenMP performance $\rightarrow$ Learn from empirical metrics $\rightarrow$ Recommend optimal configuration $\rightarrow$ Execute C binary and validate."*
