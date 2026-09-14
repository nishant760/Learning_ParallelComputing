# Presentation Synopsis & PPT Slide Structure
## AI-Assisted OpenMP Scheduling for Parallel Numerical Computation

---

### Project Synopsis

**Title**: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation  
**Domain**: Parallel Computing & Machine Learning  
**Core Technologies**: C, OpenMP, Python, Scikit-Learn, Pandas, Matplotlib, Streamlit  
**Summary**: This project addresses the challenge of optimizing loop scheduling and thread allocation in multi-threaded parallel numerical computations. Using midpoint numerical integration as the benchmark workload, the framework combines compiled C/OpenMP execution with a Random Forest Regressor trained on empirical benchmark metrics (1 warm-up + 5 measured runs taking median time) to predict speedup and recommend optimal parameters for unseen workloads.

---

### Slide 1: Title & Project Credentials
- **Title**: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation
- **Subtitle**: Workload-Aware Performance Prediction & Loop Parameter Optimization
- **Presenter**: Student Name / Roll Number
- **Department**: Department of Computer Science & Engineering

---

### Slide 2: Problem Statement & Motivation
- **Context**: High-precision numerical computing requires evaluating millions of loop iterations.
- **Problem**: Sequential single-threaded execution is bottlenecked by single-core speed.
- **OpenMP Solution**: Multi-threading divides loop iterations across physical CPU cores.
- **Challenge**: Parallel speedup depends non-linearly on thread count, scheduling policy (`static`, `dynamic`, `guided`), and chunk size. Suboptimal choices lead to scheduling overhead or load imbalance.
- **Goal**: Develop an intelligent AI-assisted framework that predicts performance and recommends optimal settings for unseen workloads.

---

### Slide 3: Computational Workload — Midpoint Integration of $\pi$
- **Formula**:
  $$\pi = \int_0^1 \frac{4}{1+x^2} \, dx \approx h \sum_{i=0}^{N-1} \frac{4}{1 + x_i^2}, \quad h = \frac{1}{N}, \ x_i = (i + 0.5)h$$
- **Characteristics**:
  - Independent iteration calculations (embarrassingly parallel).
  - Race condition on global `sum` avoided via `#pragma omp parallel for reduction(+:sum)`.
  - Serves as the benchmark workload for OpenMP performance tuning.

---

### Slide 4: System Architecture & Data Flow
- **Offline Training Pipeline**:
  - `src/benchmark.c` (1 warm-up + 5 measured runs $\to$ Median Time)
  - `data/performance.csv` (Empirical dataset: median time, mean, stddev, speedup, efficiency)
  - `ai/train_model.py` (Random Forest Regressor trained via GroupKFold + $N=75\text{M}$ holdout)
  - `model/speedup_model.pkl` (Saved Scikit-Learn pipeline)
- **Online Recommendation & Validation Pipeline**:
  - User Target Workload $N \to$ Candidate Evaluation (27 configs) $\to$ Top Recommendation $\to$ Live C Execution $\to$ Predicted vs Measured Performance.

---

### Slide 5: OpenMP Loop Scheduling Analysis
- **Static (`static, chunk`)**: Assigns equal iteration blocks at start time. Zero runtime queue locking overhead; ideal for uniform workloads.
- **Dynamic (`dynamic, chunk`)**: Iterations assigned from runtime queue. Small chunk sizes (100) introduce runtime scheduling overhead as threads request additional chunks.
- **Guided (`guided, chunk`)**: Exponentially shrinking chunk sizes. Minimizes queue locking while maintaining dynamic load balance. Guided scheduling performed best for the tested workload/configuration on the benchmark system.

---

### Slide 6: Empirical Benchmarking Methodology
- **Speedup Formula**: $S = \frac{T_{1,\text{median}}}{T_{p,\text{median}}}$
- **Efficiency Formula**: $E = \frac{S}{P} \times 100\%$
- **Noise Mitigation**: 1 warm-up run + 5 measured runs per configuration taking median execution time.
- **Floating-Point Error**: $|\pi_{\text{calculated}} - 3.141592653589793| < 10^{-14}$ (numerically accurate result with negligible floating-point error).

---

### Slide 7: AI Model Training & Evaluation
- **Model**: Random Forest Regressor (100 decision trees).
- **Features**: `N`, `threads` (2, 4, 8), `schedule` (one-hot), `chunk`.
- **Validation**:
  - Workload GroupKFold Cross-Validation: Mean $R^2 = 0.9845$, MAPE = $3.12\%$.
  - Unseen Workload Holdout ($N = 75\text{M}$): $R^2 = 0.9992$, MAE = $0.0275$, MAPE = $0.82\%$.

---

### Slide 8: Experimental Performance Graphs
- **Graph 1**: Median Execution Time vs. Threads
- **Graph 2**: Speedup vs. Threads (Median Baseline)
- **Graph 3**: Parallel Efficiency vs. Threads
- **Graph 4**: OpenMP Scheduling Policy Comparison
- **Graph 5**: Workload Scaling Across Problem Sizes
- **Graph 6**: Predicted vs. Measured Speedup

---

### Slide 9: Demonstration & User Interface
- Professional 4-Section Academic Dashboard (`app.py`):
  1. Project Overview & System Info (CPU Cores, Python Version, Hardware Note).
  2. Live Experiment (Single run parameters & Quick Demo $N=50\text{M}$).
  3. Schedule Comparison (Live Static vs Dynamic vs Guided tool).
  4. AI Recommendation (Full candidate ranking table).

---

### Slide 10: Conclusion & Limitations
- **Conclusion**:
  - Successfully built a workload-aware OpenMP performance tuning framework.
  - Random Forest regression predicts speedups on unseen workloads with $< 1\%$ error.
- **Limitations**:
  - Benchmark performance metrics are hardware-dependent.
  - Predictions depend on empirical training data scope.
  - Future work includes hardware performance counter integration and continuous online learning.
