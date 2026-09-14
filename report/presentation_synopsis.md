# Presentation Synopsis & Slide Structure
## AI-Assisted Parallel Numerical Computation and Scheduling

---

### Project & Team Details (Slide 1)
- **Title**: AI-Assisted Parallel Numerical Computation and Scheduling
- **Institution**: The National Institute of Engineering (NIE), Mysuru
- **Department**: Department of Computer Science & Engineering
- **Project Team**:
  - **4NI23CS** — Nishant
  - **4NI23CS172** — Ritika Rajesh Naik
  - **4NI24CS** — Farhan
- **Batch**: C16
- **Date**: 14-09-2026

---

### Table of Contents (Slide 2)
1. Introduction
2. Objectives
3. Existing System & Drawbacks
4. Proposed System & Advantages
5. Literature Survey
6. System Requirements Specification (SRS)
7. High-Level System Design
8. Low-Level System Design
9. System Workflow & Modules
10. Conclusion

---

### Slide-by-Slide Outline

#### Slide 3: 1. Introduction
- Large-scale numerical computations may require millions or billions of iterations and can become time-consuming on a single processor core.
- Parallel computing divides a computational workload among multiple processing units so independent operations execute simultaneously.
- Numerical integration ($\pi$ midpoint rule) is selected as the computational problem because its iterations are embarrassingly parallel and suitable for data parallelism.
- Implements a sequential baseline and parallel C programs using OpenMP.
- Static, dynamic, and guided scheduling strategies are evaluated across workloads and thread counts.
- Machine learning is added to predict execution/burst time and expected speedup for unseen workloads, supporting AI-assisted scheduling decisions.

#### Slide 4: 2. Objectives
- Develop a sequential C implementation for numerical integration.
- Develop a parallel C implementation using OpenMP and divide workload among threads.
- Compare static, dynamic, and guided scheduling strategies across chunk sizes.
- Test different input sizes ($N$) and numbers of worker threads (1, 2, 4, 8).
- Measure execution/burst time, speedup, parallel efficiency, and CPU utilization.
- Generate empirical benchmark dataset (1 warm-up + 5 measured runs using median execution time) and train a regression model.
- Predict expected speedup for unseen input sizes and compare with actual live execution results.
- Support AI-assisted scheduling decisions and optimize parallel execution.

#### Slide 5: 3. Existing System & Drawbacks
- **Existing System**:
  - Numerical computation executed sequentially using a single CPU thread.
  - Fixed parallel configuration selected without considering workload characteristics.
  - Basic parallel programs use only one scheduling policy.
  - Performance evaluated using execution time alone.
  - Schedulers do not use historical performance data to predict workload cost.
- **Drawbacks**:
  - Sequential execution becomes slow for large input sizes.
  - Equal work distribution does not always guarantee good load balance.
  - Scheduling overhead reduces parallel performance.
  - Increasing thread counts does not guarantee proportional speedup (Amdahl's Law).
  - Risk of selecting suboptimal thread counts or scheduling strategies.

#### Slide 6: 4. Proposed System & Advantages
- **Proposed System**:
  - Accept numerical workload ($N$) and configure OpenMP threads.
  - Execute sequential C baseline and measure baseline time ($T_1$).
  - Execute parallel C/OpenMP versions using static, dynamic, and guided scheduling.
  - Record execution/burst time, speedup, efficiency, and CPU utilization.
  - Train Random Forest regression model on benchmark data to predict task burst time and speedup.
  - For unseen workloads, recommend effective scheduling configurations based on ML predictions.
- **Advantages**:
  - Reduced execution time • Better load balancing • Quantitative speedup analysis • Comparison of scheduling policies • AI-assisted performance prediction • Adaptive scheduling decision.

#### Slide 7: 6. System Requirements Specification (Software Requirements)
- **Programming Language**: C
- **Parallel Framework**: OpenMP
- **Compiler**: GCC / MinGW with OpenMP support (`-O3 -fopenmp`)
- **Operating System**: macOS, Windows, or Linux
- **Development Tool**: VS Code / Terminal / Makefile
- **Data**: CSV benchmark results (`data/performance.csv`)
- **AI**: Python, Pandas, NumPy, Scikit-learn
- **Visualization & UI**: Python, Matplotlib, Streamlit
- **Version Control**: Git & GitHub

#### Slide 8: 6. SRS — Functional & Non-Functional Requirements
- **Functional Requirements**:
  - Accept input size $N$ and worker threads.
  - Execute sequential and OpenMP parallel computation.
  - Support static, dynamic, and guided scheduling.
  - Record execution time for every configuration.
  - Calculate speedup, efficiency, and utilization.
  - Generate benchmark CSV data.
  - Train ML regression model.
  - Predict performance for unseen workloads.
  - Recommend effective scheduling configuration.
- **Non-Functional Requirements**:
  - Performance, Scalability, Reliability, Accuracy, Usability, Maintainability, Reproducibility.

#### Slide 9: 7. System Design — High-Level Design
- Data Flow: `Input Workload` $\rightarrow$ `C/OpenMP Engine` $\rightarrow$ `Scheduler / Benchmark` $\rightarrow$ `Performance Metrics` $\rightarrow$ `ML Prediction` $\rightarrow$ `AI-Assisted Scheduling` $\rightarrow$ `CSV Benchmark Data`.

#### Slide 10: 8. System Design — Low-Level Design (12-Step Process)
1. Read workload size N
2. Initialize numerical integration parameters
3. Execute sequential baseline
4. Create OpenMP parallel region
5. Select scheduling strategy
6. Distribute loop iterations among threads
7. Reduce partial sums
8. Record execution time
9. Calculate speedup and efficiency
10. Generate benchmark dataset
11. Predict burst time / speedup using ML
12. Select recommended configuration

#### Slide 11: 9. System Workflow & Modules
- **Input Module**: Workload size, thread count, and scheduling parameters.
- **Benchmark Preparation**: Generate repeated runs (1 warmup + 5 measured) and store performance data.
- **Parallel Module**: Sequential + OpenMP execution; Static/Dynamic/Guided schedules.
- **Performance Module**: Time, speedup, efficiency, and utilization.
- **AI Module**: Train regression model and predict unseen workload.
- **Comparison & Visualization**: Graphs (time, speedup, efficiency, utilization, predicted vs actual) & Streamlit UI dashboard.

#### Slide 12: 10. Conclusion
- Proposed system demonstrates large-scale numerical computation using sequential C and parallel C/OpenMP implementations.
- Static, dynamic, and guided scheduling strategies evaluated across workloads and thread counts.
- Execution time, speedup, efficiency, and utilization provide quantitative measures of parallel performance.
- ML regression model predicts burst time and expected speedup for unseen workloads.
- AI-assisted scheduler recommends suitable parallel configurations.
- Demonstrates how machine learning complements traditional parallel scheduling and performance optimization.
