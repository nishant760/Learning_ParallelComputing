# Academic Project Report
## AI-Assisted Parallel Numerical Computation and Scheduling

---

### 1. Introduction

Large-scale numerical computations require millions to billions of iterations and can become computationally bottlenecked on a single processor core. Parallel computing divides a computational workload among multiple physical processing units so independent operations execute simultaneously. Numerical integration using the midpoint rule is selected as the computational workload because its iterations are embarrassingly parallel (zero loop-carried data dependencies) and highly suitable for data parallelism.

This project implements a single-threaded C sequential baseline and parallel multi-threaded C programs using OpenMP. Static, dynamic, and guided scheduling strategies are evaluated across different workloads and numbers of worker threads. Machine learning (Random Forest Regression) is incorporated to predict task execution/burst time and expected speedup for unseen workloads, supporting an AI-assisted scheduling decision.

---

### 2. Objectives

- Develop a sequential C implementation for a computationally intensive numerical integration problem ($\pi$ calculation via midpoint rule).
- Develop a parallel implementation using OpenMP and divide the workload among multiple threads using `#pragma omp parallel for reduction(+:sum)`.
- Compare static, dynamic, and guided scheduling strategies across different chunk sizes.
- Test different input sizes ($N$) and numbers of worker threads (1, 2, 4, 8).
- Measure execution/burst time, speedup, parallel efficiency, and CPU utilization.
- Generate empirical benchmark data (1 warm-up + 5 measured runs using median execution time) and train a Random Forest regression model for performance and burst-time prediction.
- Predict expected speedup for an unseen input size and compare it with actual live execution results.
- Use predictions to support AI-assisted scheduling decisions and optimize parallel execution.

---

### 3. Existing System & Drawbacks

#### Existing System
- Numerical computation is commonly executed sequentially using a single CPU thread.
- A fixed parallel configuration may be selected without considering workload characteristics.
- Basic parallel programs may use only one static scheduling policy.
- Performance is often evaluated using execution time alone.
- Conventional schedulers do not use historical performance data to predict workload cost.

#### Drawbacks
- Sequential execution becomes slow for very large input sizes ($N \ge 100,000,000$).
- Equal work distribution may not always provide good load balance across dynamic environments.
- Scheduling overhead (e.g. dynamic queue locking with small chunk sizes) can severely reduce parallel performance.
- Increasing the number of threads does not guarantee proportional speedup due to Amdahl's Law and synchronization bottlenecks.
- Without prediction, the scheduler may select a suboptimal thread count or scheduling strategy.

---

### 4. Proposed System & Advantages

#### Proposed System
- Accept a large numerical workload ($N$) and configure OpenMP worker threads.
- Execute a sequential C baseline and measure reference execution time ($T_1$).
- Execute parallel C/OpenMP versions using static, dynamic, and guided scheduling policies.
- Record median execution time, speedup, parallel efficiency, and CPU utilization for different workloads and thread counts.
- Train a Random Forest regression model using empirical benchmark data to predict task burst time and expected speedup.
- For an unseen workload, use model predictions to recommend an effective parallel scheduling configuration.

#### Key Advantages
- **Reduced Execution Time**: Accelerates large numerical integration computations using multi-core parallelism.
- **Better Load Balancing**: Dynamically evaluates static, dynamic, and guided schedules to prevent core idling.
- **Quantitative Speedup Analysis**: Evaluates speedup ($S = T_1 / T_p$) and efficiency ($E = S / P \times 100\%$) based on median execution timing.
- **Comparison of Scheduling Policies**: Demonstrates queue locking overhead and load balancing trade-offs across chunk sizes.
- **AI-Assisted Performance Prediction**: Uses machine learning regression to model non-linear CPU performance trends.
- **Adaptive Scheduling Decision**: Recommends optimal thread count, schedule strategy, and chunk size before execution.

---

### 5. Literature Survey

1. **Chandra et al. (2001) — *Parallel Programming in OpenMP***:
   Demonstrated shared-memory multi-threading techniques using OpenMP pragma directives. Established foundational principles for loop iteration splitting, reductions, and static vs dynamic scheduling overheads.
2. **Dagum & Menon (1998) — *OpenMP: An Industry Standard API for Shared-Memory Programming***:
   Explored thread synchronization costs, work-sharing constructs, and cache coherence behavior in multi-core architectures.
3. **Amdahl (1967) — *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities***:
   Formulated Amdahl's Law, defining theoretical speedup bounds based on sequential program fractions.
4. **Machine Learning for Auto-Tuning (Modern Approaches)**:
   Recent research highlights using decision trees and Random Forest regressors to predict execution time and optimal parameters for high-performance computing (HPC) loops without exhaustive runtime searching.

---

### 6. System Requirements Specification (SRS)

#### Software Requirements
- **Programming Language**: C
- **Parallel Framework**: OpenMP
- **Compiler**: GCC / MinGW with OpenMP support (`-O3 -fopenmp`)
- **Operating System**: macOS, Linux, or Windows
- **Development Tools**: VS Code / Terminal / Makefile
- **Benchmark Data Format**: CSV (`data/performance.csv`)
- **AI & ML Libraries**: Python 3.10+, Pandas, NumPy, Scikit-learn, Joblib
- **Visualization & UI**: Matplotlib, Streamlit
- **Version Control**: Git & GitHub

#### Functional Requirements
- Accept input size $N$ and number of worker threads (1, 2, 4, 8).
- Execute sequential C baseline and OpenMP parallel computations.
- Support static, dynamic, and guided scheduling policies.
- Record median execution time across 5 measured runs (after 1 warm-up run).
- Calculate speedup, parallel efficiency, CPU utilization, and floating-point error.
- Generate empirical benchmark dataset (`data/performance.csv`).
- Train a Random Forest ML regression model.
- Predict task burst-time and speedup for unseen workloads ($N = 75\text{M}$).
- Recommend an effective scheduling configuration and validate against live C execution.

#### Non-Functional Requirements
- **Performance**: Reduce execution time for large workload computations.
- **Scalability**: Support scaling input sizes ($N = 1\text{M}$ to $100\text{M}$) and thread counts (1 to 8).
- **Reliability**: Produce consistent numerical results ($\pi \approx 3.141592653589793$).
- **Accuracy**: Maintain negligible floating-point error ($< 10^{-13}$).
- **Usability**: Simple command-line interface and interactive Streamlit web dashboard.
- **Maintainability**: Modular separation of C computation, benchmarking, ML model, and UI layers.
- **Reproducibility**: Store experimental results in CSV format for verifiable comparison.

---

### 7. High-Level System Design

```text
+-----------------------+     +-----------------------+     +-----------------------+
|  User / Researcher    | --> |  C/OpenMP Parallel    | --> |     Scheduler /       |
|                       |     |       Engine          |     |      Benchmark        |
+-----------------------+     +-----------------------+     +-----------+-----------+
                                                                        |
                                                                        v
+-----------------------+     +-----------------------+     +-----------+-----------+
| CSV Benchmark Data    | <-- |    AI Model           | <-- | Performance Metrics   |
| (data/performance.csv)|     | (Burst-Time/Speedup)  |     | (Time, Speedup, Eff)  |
+-----------------------+     +-----------------------+     +-----------------------+

Data Flow: Input Workload -> C/OpenMP Engine -> Scheduler -> Performance Metrics -> ML Prediction -> AI-Assisted Scheduling
```

---

### 8. Low-Level System Design (12-Step Process)

```text
 [1. Read workload size N]
            │
            v
 [2. Initialize numerical integration parameters]
            │
            v
 [3. Execute sequential baseline (T1)]
            │
            v
 [4. Create OpenMP parallel region]
            │
            v
 [5. Select scheduling strategy (Static/Dynamic/Guided)]
            │
            v
 [6. Distribute loop iterations among threads]
            │
            v
 [7. Reduce partial sums (#pragma omp reduction)]
            │
            v
 [8. Record median execution time]
            │
            v
 [9. Calculate speedup and efficiency]
            │
            v
 [10. Generate benchmark dataset (data/performance.csv)]
            │
            v
 [11. Predict burst time / speedup using ML]
            │
            v
 [12. Select recommended configuration & validate]
```

---

### 9. System Workflow & Modules

1. **Input Module**: Accepts workload size $N$, thread count (1, 2, 4, 8), and scheduling parameters.
2. **Benchmark Preparation**: Generates repeated runs (1 warm-up + 5 measured runs) and stores empirical performance data into CSV.
3. **Parallel Module**: Compiles and executes sequential and OpenMP parallel C binaries using Static, Dynamic, and Guided scheduling.
4. **Performance Module**: Calculates median execution time, speedup ($S = T_1 / T_p$), parallel efficiency ($E = S / P \times 100\%$), CPU utilization, and floating-point error.
5. **AI Module**: Trains Random Forest Regression model on benchmark data and predicts burst-time / speedup for unseen workloads ($N = 75\text{M}$).
6. **Comparison & Visualization**: Generates matplotlib performance charts (time, speedup, efficiency, schedule comparison, predicted vs measured speedup) and renders Streamlit UI dashboard.

---

### 10. Limitations & Conclusion

#### Limitations
- **Hardware Dependence**: Empirical benchmark metrics depend strictly on host CPU architecture, physical core counts, and memory bandwidth.
- **Training Data Scope**: Predictions apply to parameter ranges represented in training data ($N \in [1\text{M}, 100\text{M}]$, Threads $\in [2, 8]$).
- **Environment Scope**: Tests unseen iteration counts on the current machine architecture. A multi-node cluster scheduler would require hardware counters and continuous online learning.

#### Conclusion
The proposed system demonstrates large-scale numerical computation using sequential C and parallel C/OpenMP implementations. Static, dynamic, and guided scheduling strategies are evaluated across multiple workloads and thread counts. Execution time, speedup, parallel efficiency, and CPU utilization provide quantitative measures of parallel performance. A Random Forest regression model predicts task burst-time and expected speedup for unseen workloads, enabling an AI-assisted scheduler to recommend suitable parallel configurations and complement traditional performance optimization.
