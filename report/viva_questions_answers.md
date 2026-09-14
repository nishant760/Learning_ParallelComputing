# Viva Voce Questions & Answers
## AI-Assisted OpenMP Scheduling for Parallel Numerical Computation

---

### Q1: What is the mathematical problem being solved in this project?
**Answer:**
We evaluate the definite integral:
$$\pi = \int_0^1 \frac{4}{1+x^2} \, dx$$
using numerical integration via the **midpoint rule**. The interval $[0, 1]$ is divided into $N$ equal sub-intervals of width $h = 1/N$. The height of each rectangle is evaluated at the midpoint $x_i = (i + 0.5)h$. Summing the rectangle areas approximates $\pi$. Note that $\pi$ calculation serves as the benchmark workload used to study OpenMP loop performance tuning.

---

### Q2: Why is numerical integration suitable for parallel computation?
**Answer:**
Numerical integration using the midpoint rule is **embarrassingly parallel**. Each sub-interval evaluation $f(x_i) = \frac{4}{1+x_i^2}$ is completely independent of all other iterations. There are no loop-carried data dependencies.

---

### Q3: What is OpenMP and how does it work in C?
**Answer:**
OpenMP (Open Multi-Processing) is an API based on compiler directives (`#pragma omp`), library routines, and environment variables for shared-memory parallel programming in C/C++. It follows the **fork-join model**: execution starts on a master thread; when encountering `#pragma omp parallel`, the master thread forks worker threads to execute parallel loop iterations, joining them upon completion.

---

### Q4: Why is the `reduction(+:sum)` clause critical in your parallel C code?
**Answer:**
Without reduction, multiple OpenMP threads would simultaneously attempt to write to the shared variable `sum`, creating a **race condition**.

The `#pragma omp reduction(+:sum)` clause instructs OpenMP to create thread-private partial accumulation variables initialized to 0. During loop execution, threads accumulate local results into private copies without locking overhead. At loop exit, OpenMP combines private partial results into the global `sum` using tree-based reduction.

---

### Q5: How was the empirical benchmarking methodology improved in this project?
**Answer:**
Instead of taking the minimum execution time from 3 runs (which biases results toward optimistic values), our benchmark methodology performs **1 warm-up run + 5 measured runs per configuration** and uses the **MEDIAN execution time** as the primary timing metric. We also calculate mean, standard deviation, minimum time, and floating-point error to eliminate measurement noise.

---

### Q6: Explain the difference between `static`, `dynamic`, and `guided` OpenMP scheduling policies.
**Answer:**
1. **Static (`schedule(static, chunk)`)**: Loop iterations are divided into fixed blocks of size `chunk` and assigned to threads at loop start in round-robin order. Incurs zero runtime queue locking overhead; ideal for uniform workloads.
2. **Dynamic (`schedule(dynamic, chunk)`)**: Chunks of size `chunk` are placed in a runtime queue. Threads dynamically request additional chunks as they finish. Small chunk sizes (e.g. 100) introduce runtime scheduling overhead as threads frequently request additional chunks from the shared queue.
3. **Guided (`schedule(guided, chunk)`)**: Similar to dynamic, but chunk sizes start large and exponentially shrink down to `chunk` as loop completion approaches. Minimizes initial queue locking overhead while maintaining dynamic load balancing near loop completion. Guided scheduling performed best for the tested workload/configuration on the benchmark system.

---

### Q7: What is Speedup ($S$) and how is it calculated in your benchmark?
**Answer:**
Speedup measures the performance gain of parallel execution relative to single-threaded sequential execution:
$$S = \frac{T_{1,\text{median}}}{T_{p,\text{median}}}$$
where $T_{1,\text{median}}$ is the median single-threaded sequential execution time and $T_{p,\text{median}}$ is the median multi-threaded execution time.

---

### Q8: What is Parallel Efficiency ($E$)?
**Answer:**
Parallel Efficiency measures core utilization:
$$E = \frac{S}{P} \times 100\%$$
where $P$ is the number of OpenMP threads.

---

### Q9: What is Amdahl's Law and what does it state?
**Answer:**
Amdahl's Law defines the maximum theoretical speedup achievable by parallelizing a program with sequential fraction $f$:
$$S_{\text{max}}(P) = \frac{1}{f + \frac{1 - f}{P}}$$
As $P \to \infty$, $S_{\text{max}} \to \frac{1}{f}$.

---

### Q10: What role does machine learning play in this project?
**Answer:**
Parallel performance depends non-linearly on input size ($N$), thread count, schedule, and chunk size. Instead of relying on static rules or manual trial-and-error testing, we train a **Random Forest Regressor** on empirical benchmark data to predict speedup for unseen workloads and recommend optimal configurations.

---

### Q11: What regression error metrics are used to report model performance?
**Answer:**
We report regression performance using:
- **$R^2$ Determination Score**: Measures proportion of variance explained by model (e.g. $0.9992$).
- **MAE (Mean Absolute Error)**: Average absolute error in predicted speedup units.
- **RMSE (Root Mean Squared Error)**: Standard deviation of prediction residuals.
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error relative to measured speedup (e.g. $< 1\%$).

---

### Q12: How was data leakage avoided during machine learning cross-validation?
**Answer:**
We implemented **GroupKFold cross-validation** grouped by workload size $N$. This ensures all benchmark rows belonging to a specific workload size $N$ are held out together, preventing rows from the same workload size from appearing in both training and validation folds.

---

### Q13: How does `ai_scheduler.py` work?
**Answer:**
Given a target workload $N$:
1. It constructs 27 candidate configurations ($\text{schedule} \times \text{threads} [2,4,8] \times \text{chunk}$).
2. Uses the saved Random Forest model (`speedup_model.pkl`) to predict speedup for all candidates.
3. Ranks candidates and selects the top recommendation.
4. Triggers live execution of compiled C binaries (`bin/sequential_pi` and `bin/parallel_pi`).
5. Displays a live comparison of **Predicted Performance** vs. **Measured Performance** and calculates prediction error.

---

### Q14: Does the Random Forest model directly control the CPU or thread scheduling?
**Answer:**
No. The Random Forest model is an offline-trained regression predictor that recommends optimal execution configuration parameters. The C compiled program and the OpenMP runtime system perform the physical execution on the CPU.

---

### Q15: What compiler flags were used to compile the OpenMP C programs?
**Answer:**
- `-O3`: Maximum level compiler optimization.
- `-fopenmp`: Enables OpenMP pragma parsing and links runtime library (`libomp`).
- `-Wall`: Enables compiler diagnostic warnings.
- `-lm`: Links standard math library.

Both sequential and parallel binaries are compiled under identical optimization rules (`-O3 -fopenmp`).

---

### Q16: What are the main limitations of this project?
**Answer:**
1. Benchmark performance numbers are strictly hardware-dependent.
2. Machine learning predictions depend on the coverage and quality of empirical training data.
3. The model predicts within the tested configuration space ($N \in [1\text{M}, 100\text{M}]$, Threads $\in [2, 8]$).
4. A enterprise-grade scheduler would require hardware performance counter features (LLC misses, memory bandwidth) and continuous online retraining.
