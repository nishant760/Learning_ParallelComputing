# Viva Voce Questions & Answers
## AI-Assisted Parallel Numerical Computation & Scheduling Using C/OpenMP

---

### Q1: What is the mathematical problem being solved in this project?
**Answer:**
We are evaluating the definite integral:
$$\pi = \int_0^1 \frac{4}{1+x^2} \, dx$$
using numerical integration via the **midpoint rule** (rectangle method). The interval $[0, 1]$ is divided into $N$ equal sub-intervals of width $h = 1/N$. The height of each rectangle is evaluated at the midpoint $x_i = (i + 0.5)h$. Summing the rectangle areas approximates $\pi$.

---

### Q2: Why is numerical integration suitable for parallel computation?
**Answer:**
Numerical integration using the midpoint rule is **embarrassingly parallel**. Each iteration $i$ computes $f(x_i) = \frac{4}{1+x_i^2}$ independently of all other iterations. There are no loop-carried data dependencies.

---

### Q3: What is OpenMP and how does it work in C?
**Answer:**
OpenMP (Open Multi-Processing) is an API based on compiler directives (`#pragma omp`), library routines, and environment variables for shared-memory parallel programming in C, C++, and Fortran. It employs a **fork-join model**: the master thread forks worker threads when encountering a parallel region and joins them upon completion.

---

### Q4: Why is the `reduction(+:sum)` clause critical in your parallel C code?
**Answer:**
Without the `reduction` clause, multiple OpenMP threads would simultaneously attempt to write to the shared variable `sum`, creating a **race condition** and producing incorrect mathematical results. 

The `reduction(+:sum)` clause instructs OpenMP to create a thread-private copy of `sum` for each thread initialized to 0. During loop execution, threads accumulate local results into their private copies without locking overhead. At loop exit, OpenMP combines private sums into the global `sum` using tree-based reduction.

---

### Q5: How do you set thread counts dynamically in OpenMP?
**Answer:**
In C code:
```c
#include <omp.h>
omp_set_num_threads(num_threads);
```
Alternatively, at runtime via shell environment variable: `export OMP_NUM_THREADS=4`.

---

### Q6: Explain the difference between `static`, `dynamic`, and `guided` OpenMP scheduling policies.
**Answer:**
1. **Static (`schedule(static, chunk)`)**: Loop iterations are divided into fixed blocks of size `chunk` and assigned to threads statically at loop start in round-robin order. Has the lowest scheduling overhead.
2. **Dynamic (`schedule(dynamic, chunk)`)**: Chunks of size `chunk` are placed in a shared task queue. As threads finish work, they dynamically request the next chunk from the queue. Excellent for irregular/unbalanced workloads, but introduces synchronization overhead.
3. **Guided (`schedule(guided, chunk)`)**: Similar to dynamic, but chunk sizes start large and decrease exponentially down to `chunk` as remaining iterations decrease. Reduces initial queue contention while maintaining load balance near loop completion.

---

### Q7: What happened when dynamic scheduling was tested with a small chunk size (`chunk=100`) on 8 threads?
**Answer:**
With $N = 100\text{M}$ and `chunk=100`, there are $1,000,000$ chunk requests sent to the shared task queue. The frequent mutex locking/unlocking overhead dominated computation time, causing speedup to collapse to **1.70x** (Efficiency 21.23%), whereas `static` and `guided` schedules achieved **5.85x** speedup.

---

### Q8: What is Speedup ($S$) and how is it calculated?
**Answer:**
Speedup measures the performance gain of parallel execution relative to sequential execution:
$$S = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$$
where $T_{\text{sequential}}$ is single-threaded execution time and $T_{\text{parallel}}$ is multi-threaded execution time.

---

### Q9: What is Parallel Efficiency ($E$)?
**Answer:**
Parallel Efficiency measures how effectively available CPU cores are utilized:
$$E = \frac{\text{Speedup}}{\text{Number of Threads } P} \times 100\% = \frac{S}{P} \times 100\%$$
An ideal parallel system has $100\%$ efficiency ($S = P$).

---

### Q10: What is Amdahl's Law and what does it state?
**Answer:**
Amdahl's Law defines the maximum theoretical speedup achievable by parallelizing a program with sequential fraction $f$:
$$S_{\text{max}}(P) = \frac{1}{f + \frac{1 - f}{P}}$$
As $P \to \infty$, $S_{\text{max}} \to \frac{1}{f}$. If $5\%$ of a program is strictly sequential ($f=0.05$), the maximum achievable speedup is 20x regardless of how many CPU cores are added.

---

### Q11: Why does parallel speedup eventually plateau or decrease as thread count increases?
**Answer:**
Due to:
1. Overhead of thread creation, synchronization, and joining.
2. Shared memory bus bandwidth saturation.
3. Cache contention and false sharing.
4. OS context switching when thread count exceeds physical core count.
5. Sequential portions of code (Amdahl's Law).

---

### Q12: How did you measure execution time accurately in C?
**Answer:**
We used OpenMP's high-resolution timer:
```c
double start_time = omp_get_wtime();
// Computation loop
double end_time = omp_get_wtime();
double elapsed = end_time - start_time;
```
`omp_get_wtime()` returns wall-clock time in seconds with microsecond resolution.

---

### Q13: What role does machine learning play in this project?
**Answer:**
Parallel performance depends non-linearly on input size ($N$), thread count, schedule, and chunk size. Instead of relying on static rules or manual testing, we train a **Random Forest Regressor** on empirical benchmark data to predict speedup for unseen workloads and select the optimal thread/schedule configuration automatically.

---

### Q14: What features are used to train the Random Forest model?
**Answer:**
1. `N`: Workload iteration count (numeric)
2. `threads`: Thread count (numeric)
3. `schedule`: OpenMP schedule policy (`static`, `dynamic`, `guided`, one-hot encoded)
4. `chunk`: Chunk size (numeric)

The target variable is `speedup` (float).

---

### Q15: How was the machine learning model evaluated on unseen workloads?
**Answer:**
We held out the entire benchmark dataset for $N = 75,000,000$ (36 combinations) during training. The Random Forest model was trained on $N \in \{1\text{M}, 10\text{M}, 20\text{M}, 50\text{M}, 100\text{M}\}$ and evaluated zero-shot on $N = 75\text{M}$. It achieved an $R^2$ score of **0.9992** and MAPE of **0.82%**.

---

### Q16: How does `ai_scheduler.py` work?
**Answer:**
Given a new target workload $N$:
1. It constructs a dataframe of all candidate execution configurations ($\text{schedule} \times \text{threads} \times \text{chunk}$).
2. Uses the saved Random Forest model (`speedup_model.pkl`) to predict speedup for all candidates.
3. Selects the configuration with the highest predicted speedup.
4. Triggers execution of compiled C binaries (`sequential_pi` and `parallel_pi`) with those parameters.
5. Displays a live comparison of predicted vs actual speedup and calculates percentage error.

---

### Q17: What is false sharing and how does OpenMP avoid it in reduction loops?
**Answer:**
False sharing occurs when threads on different cores update independent variables residing on the same L1 cache line, forcing cache invalidations across cores. OpenMP's `reduction` avoids false sharing by assigning each thread a private scalar variable allocated on thread-local stack frames or separate cache lines.

---

### Q18: What compiler flags were used to compile the OpenMP C programs?
**Answer:**
- `-O3`: Maximum level compiler optimization (loop unrolling, vectorization).
- `-fopenmp`: Enables OpenMP pragma parsing and links OpenMP runtime library (`libomp`).
- `-Wall`: Enables compiler diagnostic warnings.
- `-lm`: Links standard math library.

---

### Q19: Why was `__asm__ __volatile__("" ::: "memory")` added to `benchmark.c`?
**Answer:**
At `-O3` optimization, GCC eliminated the sequential timing loop because its return value was not printed to stdout or escaped. Adding a compiler memory barrier (`__asm__ __volatile__("" ::: "memory")`) prevents GCC from reordering or deleting the timing loop during inter-procedural optimization.

---

### Q20: What is the difference between concurrency and parallelism?
**Answer:**
- **Concurrency**: Managing multiple tasks at the same time by interleaving execution on a single core (time-slicing).
- **Parallelism**: Executing multiple tasks simultaneously on physically separate CPU cores or hardware threads.

---

### Q21: What is Gustafson's Law and how does it differ from Amdahl's Law?
**Answer:**
Amdahl's Law assumes problem size $N$ is fixed as core count increases (strong scaling). Gustafson's Law assumes problem size expands proportionally with core count (weak scaling), proving that parallel speedup scales linearly if problem size scales with processor count.

---

### Q22: Why did you choose Random Forest over Linear Regression for speedup prediction?
**Answer:**
Parallel performance exhibits non-linear threshold effects (e.g. overhead jump when chunk size drops to 100 on dynamic schedule). Linear regression cannot capture step-function thresholds or non-linear interaction terms between scheduling strategies and thread counts. Random Forest handles non-linear interactions and categorical feature splits effectively.

---

### Q23: How does the Streamlit academic UI connect with the C executables?
**Answer:**
Streamlit executes compiled C binaries (`bin/sequential_pi` and `bin/parallel_pi`) as child processes using Python's `subprocess.run()`, capturing standard output, parsing execution time and calculated $\pi$ values, and rendering metrics live on the web interface.

---

### Q24: What is the significance of the calculated $\pi$ error check?
**Answer:**
It verifies numerical correctness. For $N = 100,000,000$, both sequential and parallel calculations yield $\pi \approx 3.141592653589793$ with absolute error $< 10^{-10}$, proving that parallelization introduced zero precision degradation or race conditions.

---

### Q25: What is the primary takeaway of this project?
**Answer:**
Optimal parallel performance cannot be guaranteed solely by increasing thread counts or picking default scheduling options. Combining high-performance C/OpenMP multi-threading with Machine Learning enables intelligent, workload-adaptive scheduling that maximizes speedup and hardware efficiency.
