# Presentation Synopsis & PPT Slide Structure
## AI-Assisted Parallel Numerical Computation & Scheduling Using C/OpenMP

---

### Project Synopsis

**Title**: AI-Assisted Parallel Numerical Computation and Scheduling Using C/OpenMP  
**Domain**: Parallel Computing & Machine Learning  
**Core Technologies**: C, OpenMP, Python, Scikit-Learn, Pandas, Matplotlib, Streamlit  
**Summary**: This project addresses the challenge of optimizing loop scheduling and thread allocation in multi-threaded parallel numerical integration ($\pi$ calculation via the midpoint rule). By combining compiled C/OpenMP parallel execution with a Random Forest Regressor trained on empirical benchmark metrics, the system automatically predicts speedup and selects optimal parallel parameters for unseen workloads.

---

### Slide 1: Title & Project Credentials
- **Title**: AI-Assisted Parallel Numerical Computation and Scheduling Using C/OpenMP
- **Subtitle**: Workload-Adaptive Loop Scheduling & Performance Prediction
- **Presenter**: Student Name / Roll Number
- **Department**: Department of Computer Science & Engineering
- **Guide**: Project Supervisor / Professor Name

---

### Slide 2: Introduction & Motivation
- **Context**: High-precision numerical computing requires evaluating millions of iterations.
- **Problem**: Sequential execution is bottlenecked by single-core speed.
- **OpenMP Solution**: Multi-threading divides loop iterations across physical CPU cores.
- **Challenge**: Parallel speedup depends non-linearly on thread count, scheduling policy (`static`, `dynamic`, `guided`), and chunk size. Suboptimal choices lead to overhead penalties or load imbalance.
- **Goal**: Develop an intelligent AI-assisted framework that predicts performance and recommends optimal settings for unseen workloads.

---

### Slide 3: Core Problem — Numerical Integration of $\pi$
- **Formula**:
  $$\pi = \int_0^1 \frac{4}{1+x^2} \, dx \approx h \sum_{i=0}^{N-1} \frac{4}{1 + x_i^2}, \quad h = \frac{1}{N}, \ x_i = (i + 0.5)h$$
- **Loop Characteristics**:
  - Independent iteration calculations (no loop-carried data dependencies).
  - Race condition on global `sum` prevented via `#pragma omp parallel for reduction(+:sum)`.
  - Scalable workload ($N = 1\text{M}$ to $100\text{M}$).

---

### Slide 4: System Architecture & Data Flow
- **C/OpenMP Engine**: Compiled binaries (`sequential_pi`, `parallel_pi`, `benchmark`).
- **Empirical Dataset**: 216 empirical benchmark runs exported to `data/performance.csv`.
- **Machine Learning Model**: Random Forest Regressor (`model/speedup_model.pkl`).
- **AI Scheduler Module**: Evaluates candidates and triggers live executable comparison.
- **Academic Dashboard**: Interactive Streamlit interface (`app.py`).

---

### Slide 5: OpenMP Loop Scheduling Analysis
- **Static (`static, chunk`)**: Equal block distribution. Lowest overhead; ideal for uniform iterations.
- **Dynamic (`dynamic, chunk`)**: Task queue allocation. High overhead for small chunks (`chunk=100` drops speedup to 1.70x at 8 threads).
- **Guided (`guided, chunk`)**: Exponentially shrinking chunks. Combines dynamic load balancing with minimal queue overhead (achieves 5.85x speedup at 8 threads).

---

### Slide 6: Empirical Benchmarking & Metrics
- **Speedup Formula**: $S = \frac{T_{\text{seq}}}{T_{\text{par}}}$
- **Efficiency Formula**: $E = \frac{S}{P} \times 100\%$
- **Benchmark Matrix**:
  - Workloads: 1M, 10M, 20M, 50M, 75M, 100M
  - Threads: 1, 2, 4, 8
  - Schedules: Static, Dynamic, Guided
  - Chunks: 100, 1000, 10000

---

### Slide 7: AI Model Training & Evaluation
- **Algorithm**: Random Forest Regressor (100 decision trees).
- **Feature Matrix**: `N`, `threads`, `schedule` (one-hot), `chunk`.
- **Unseen Workload Evaluation ($N = 75\text{M}$)**:
  - $R^2$ Score: **0.9992**
  - Mean Absolute Error (MAE): **0.0275**
  - Mean Absolute Percentage Error (MAPE): **0.82%**

---

### Slide 8: Experimental Performance Graphs
- **Graph 1**: Execution Time vs. Threads (Exponential reduction from 0.088s to 0.015s).
- **Graph 2**: Speedup vs. Threads (Near-linear scaling up to 4 threads; sub-linear saturation at 8 threads).
- **Graph 3**: Efficiency vs. Threads (95.5% at 2 threads down to 73.1% at 8 threads).
- **Graph 4**: OpenMP Schedule Comparison (Guided and Static outperform Dynamic with small chunks).
- **Graph 5**: AI Predicted vs. Actual Speedup (High alignment along $y=x$ ideal line).

---

### Slide 9: Demonstration & User Interface
- Academic Streamlit Dashboard:
  - Input parameters: Workload $N$, Threads, Schedule, Chunk size.
  - Displays: Sequential Time, Parallel Time, Measured Speedup, Parallel Efficiency, $\pi$ Precision Error.
  - Live AI speedup prediction and optimal configuration recommendation.

---

### Slide 10: Conclusion & References
- **Conclusion**:
  - Successfully parallelized numerical integration using C/OpenMP with zero precision loss.
  - Demonstrated empirical load balancing and scheduling overhead trade-offs.
  - Built an AI scheduler capable of predicting speedup on unseen workloads with $< 1\%$ error.
- **Key References**:
  - OpenMP Application Program Interface, Version 5.2.
  - Chandra, R., et al. *Parallel Programming in OpenMP*, Morgan Kaufmann.
  - Scikit-Learn Documentation: Random Forest Regression.
