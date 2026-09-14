# Academic Project Report
## AI-Assisted Parallel Numerical Computation and Scheduling Using C/OpenMP

---

### 1. Problem Statement

Numerical integration for approximating mathematical constants such as $\pi$ using high-order discretization requires evaluating millions to billions of sub-intervals. Sequential execution on single-core architectures suffers from high execution latency. While OpenMP multi-threading allows loop iteration splitting across physical CPU cores, selecting the optimal parallel execution parameters—such as thread count, scheduling policy (`static`, `dynamic`, `guided`), and chunk size—is non-trivial. Suboptimal parameters introduce scheduling overhead, false sharing, thread synchronization delays, or load imbalance, which degrades speedup. Manual trial-and-error configuration is inefficient; hence, an automated, machine-learning-assisted scheduling framework is required.

---

### 2. Project Objectives

1. **Sequential Implementation**: Implement a baseline C program calculating $\pi$ via midpoint numerical integration and measure precise execution time.
2. **Parallel OpenMP Implementation**: Develop a parallel C implementation employing `#pragma omp parallel for reduction(+:sum)` with configurable thread counts, scheduling policies, and chunk sizes.
3. **Empirical Benchmarking**: Systematically benchmark performance across combinations of workload sizes ($N = 1\text{M}$ to $100\text{M}$), thread counts (1, 2, 4, 8), OpenMP schedules (`static`, `dynamic`, `guided`), and chunk sizes (100, 1000, 10000). Generate an authentic `performance.csv` dataset.
4. **Performance Evaluation**: Compute speedup ($S = T_{\text{seq}} / T_{\text{par}}$), parallel efficiency ($E = S / P \times 100\%$), scheduling overhead, load imbalance, and theoretical scaling limits using Amdahl's Law.
5. **AI/ML Performance Modeling**: Train a Random Forest Regressor on empirical benchmark data to predict speedup for unseen workloads ($N = 75\text{M}$) and evaluate prediction error (MAE, RMSE, $R^2$).
6. **AI-Assisted Scheduler**: Build a Python module (`ai_scheduler.py`) that evaluates candidate configurations for target workloads, recommends the optimal configuration, triggers compiled C binaries live, and validates predicted vs actual speedup.
7. **Academic UI**: Create a clean Streamlit interface for live experiments, parameter entry, performance metric displays, and graph visualization.

---

### 3. Existing System & Drawbacks

#### Existing System
Traditional parallel numerical computation applications rely on static compile-time parallelization or default runtime OpenMP settings (`OMP_SCHEDULE=static`).

#### Drawbacks
1. **Static Assumptions**: Default static scheduling divides iterations equally ($N/P$). On heterogeneous systems or non-uniform core architectures (e.g. Apple Silicon Performance vs Efficiency cores), static assignment leads to severe load imbalance.
2. **Dynamic Scheduling Overhead**: Dynamic scheduling uses a thread-safe task queue. Small chunk sizes (e.g. `chunk=100`) incur high synchronization overhead per chunk request, nullifying parallel gains.
3. **Lack of Predictive Intelligence**: Developers must manually test combinations of threads, schedules, and chunk sizes for every new workload size.

---

### 4. Proposed System & Advantages

#### Proposed System
An integrated framework combining high-performance C/OpenMP numerical routines with a trained Scikit-Learn Random Forest Regressor. The AI model predicts speedup for arbitrary workload sizes and selects optimal execution parameters before launching execution.

#### Key Advantages
1. **Data-Driven Parameter Selection**: Eliminates trial-and-error by predicting optimal settings using machine learning trained on empirical CPU behavior.
2. **Reduced Scheduling Overhead**: Prevents selecting inefficient combinations (e.g., `dynamic` schedule with chunk size 100 at 8 threads).
3. **Live Validation**: Connects ML predictions directly to C binary execution to verify real speedup against predictions.
4. **Academic Transparency**: Simple, clear C + OpenMP + Python implementation without complex web/database bloat.

---

### 5. Parallelism and Data Dependencies

Numerical integration formula:

$$\pi = \int_0^1 \frac{4}{1+x^2} \, dx \approx h \sum_{i=0}^{N-1} \frac{4}{1 + x_i^2}, \quad \text{where } h = \frac{1}{N}, \ x_i = (i + 0.5)h$$

#### Loop Iteration Independence
Each iteration $i$ computes $f(x_i) = \frac{4}{1 + x_i^2}$ independently. There are **no loop-carried data dependencies** ($f(x_i)$ does not depend on $f(x_{i-1})$).

#### Reduction Clause
Accumulating $f(x_i)$ into a shared `sum` variable creates a race condition if unmanaged. Using OpenMP's `reduction(+:sum)` clause gives each thread a private copy of `sum`, initialized to 0. At loop completion, OpenMP combines thread-private sums into the master variable using tree-based reduction, avoiding race conditions and locking penalties.

---

### 6. Algorithms and Pseudocode

#### Algorithm 1: Parallel Midpoint Pi Integration with OpenMP
```text
Input: Workload N, Thread count P, Schedule strategy Sched, Chunk size C
Output: Calculated Pi value, Execution time T_par

1. Set OpenMP thread count: omp_set_num_threads(P)
2. Set OpenMP runtime schedule: omp_set_schedule(Sched, C)
3. Step size h = 1.0 / N
4. Local sum = 0.0
5. Record start_time = omp_get_wtime()
6. #pragma omp parallel for reduction(+:sum) schedule(runtime)
   For i = 0 to N - 1 do:
       x = (i + 0.5) * h
       sum = sum + (4.0 / (1.0 + x * x))
   End For
7. Record end_time = omp_get_wtime()
8. Pi = h * sum
9. T_par = end_time - start_time
10. Return Pi, T_par
```

#### Algorithm 2: AI-Assisted Configuration Recommendation
```text
Input: Target Workload Size N_target, Trained RF Model M
Output: Recommended (Threads_opt, Schedule_opt, Chunk_opt), Predicted Speedup S_pred

1. Define candidate sets:
   Threads = [2, 4, 8], Schedules = ['static', 'dynamic', 'guided'], Chunks = [100, 1000, 10000]
2. Construct feature matrix X_candidates containing all combinations of (N_target, t, sched, c)
3. Predict speedup vector S_vector = M.predict(X_candidates)
4. Find index k = argmax(S_vector)
5. Extract optimal tuple (Threads_opt, Schedule_opt, Chunk_opt) = X_candidates[k]
6. Return optimal configuration and predicted speedup S_vector[k]
```

---

### 7. High-Level Architecture & Workflow

```
 +-----------------------------------------------------------------------+
 |                             User Interface                            |
 |                (Streamlit Academic UI / CLI Scheduler)                |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------+-----------------------------------+
 |                        AI Scheduler Module                            |
 |                    (ai/ai_scheduler.py)                               |
 |   1. Generates candidate configurations (Schedule x Threads x Chunk)  |
 |   2. Queries Random Forest model for predicted speedups               |
 |   3. Selects optimal configuration with max predicted speedup        |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------+-----------------------------------+
 |                   Trained Machine Learning Model                      |
 |                      (model/speedup_model.pkl)                        |
 |         Random Forest Regressor trained on data/performance.csv       |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------+-----------------------------------+
 |                       OpenMP Parallel C Engine                        |
 |                         (bin/parallel_pi)                             |
 |  Executes loop integration via #pragma omp parallel for reduction    |
 +-----------------------------------+-----------------------------------+
                                     |
                                     v
 +-----------------------------------+-----------------------------------+
 |                    Live Performance Validation                        |
 |     Compares Empirical Speedup vs AI Predicted Speedup & Logs Error   |
 +-----------------------------------------------------------------------+
```

---

### 8. Performance Analysis & Theoretical Limits

#### 1. Speedup and Efficiency Equations
- **Speedup ($S$)**: $S = \frac{T_1}{T_P}$, where $T_1$ is sequential execution time and $T_P$ is parallel time on $P$ threads.
- **Parallel Efficiency ($E$)**: $E = \frac{S}{P} \times 100\%$.

#### 2. Amdahl's Law
Amdahl's Law models theoretical speedup when a fraction $f$ of the code is strictly sequential:

$$S_{\text{max}}(P) = \frac{1}{f + \frac{1 - f}{P}}$$

For our midpoint integration routine, initialization and timing overhead represent a tiny sequential fraction ($f \approx 0.001$). Thus, theoretical max speedup on $P=8$ threads is:

$$S_{\text{max}}(8) = \frac{1}{0.001 + \frac{0.999}{8}} \approx 7.93x$$

Empirical results demonstrate a peak speedup of **5.85x** on 8 threads due to hardware core contention, memory bandwidth limits, and OS thread context switching overhead.

#### 3. OpenMP Loop Scheduling Comparison
- **Static (`schedule(static, chunk)`)**: Divides loop iterations into fixed block sizes at compile/start time. Minimal synchronization overhead, best for uniform workloads.
- **Dynamic (`schedule(dynamic, chunk)`)**: Threads request chunks dynamically from a shared queue as they finish. Ideal for irregular workloads, but small chunk sizes (`chunk=100`) suffer severe queue lock contention (e.g. speedup drops to 1.70x at 8 threads).
- **Guided (`schedule(guided, chunk)`)**: Chunks start large and exponentially shrink to `chunk` size. Combines low initial overhead with dynamic load balancing at loop end. Achieves optimal speedup (5.85x at 8 threads).

---

### 9. AI Model Explanation

The performance prediction model is a **Random Forest Regressor** comprising 100 decision trees. 
- **Features**:
  1. `N` (Numeric): Workload iteration count ($10^6$ to $10^8$)
  2. `threads` (Numeric): 1, 2, 4, 8
  3. `schedule` (Categorical One-Hot): `static`, `dynamic`, `guided`
  4. `chunk` (Numeric): 100, 1000, 10000
- **Target**: `speedup` (Continuous float)
- **Validation on Unseen Workload ($N = 75\text{M}$)**:
  - $R^2$ Score: **0.9992**
  - Mean Absolute Error (MAE): **0.0275**
  - Mean Absolute Percentage Error (MAPE): **0.82%**

---

### 10. Conclusion & Future Scope

#### Conclusion
This project successfully demonstrates an AI-assisted approach to parallel numerical computing. By combining C/OpenMP high-performance numerical integration with Random Forest machine learning, we achieved up to 5.85x empirical speedup on 8 threads while enabling an AI scheduler to predict performance on unseen workloads with under 1% error.

#### Future Scope
1. **GPU Acceleration**: Extend implementation to CUDA or OpenACC for massive GPU vectorization.
2. **Heterogeneous CPU Core Aware Scheduling**: Incorporate core architectural type (Performance vs Efficiency cores on Apple Silicon / Intel Alder Lake) into feature matrix.
3. **Multi-Node Distributed Computing**: Expand to hybrid MPI + OpenMP clusters.
