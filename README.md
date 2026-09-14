# AI-Assisted Parallel Numerical Computation and Scheduling Using C/OpenMP

A student-developed parallel computing project for numerical integration ($\pi$ calculation) using C, OpenMP, and Machine Learning (Random Forest) to predict and recommend optimal thread and loop scheduling configurations.

---

## 📌 Project Overview

Numerical integration for calculating $\pi$ via the midpoint rule:

$$\pi = \int_0^1 \frac{4}{1+x^2} \, dx$$

is computationally intensive for large iteration counts ($N \ge 100,000,000$). This project parallelizes the computation across multi-core processors using OpenMP in C. Because parallel performance depends non-linearly on thread counts, loop scheduling strategies (`static`, `dynamic`, `guided`), and chunk sizes, we train a **Random Forest Regressor** on empirical benchmark data to predict speedup and automatically recommend optimal thread and scheduling parameters for unseen workloads.

---

## 📁 Repository Structure

```
ParallelComputing/
├── src/
│   ├── sequential_pi.c   # Baseline single-threaded C implementation
│   ├── parallel_pi.c     # Multi-threaded OpenMP C implementation with reduction
│   └── benchmark.c       # Empirical benchmarking suite for performance dataset
├── ai/
│   ├── train_model.py    # Random Forest training on benchmark dataset
│   ├── ai_scheduler.py   # AI scheduler and live prediction-vs-actual validator
│   └── visualize.py      # Matplotlib performance plot generator
├── data/
│   └── performance.csv   # Real empirical benchmark dataset (N, threads, schedule, chunk, etc.)
├── model/
│   └── speedup_model.pkl # Trained Scikit-Learn Random Forest pipeline
├── results/              # Generated high-resolution performance plots
│   ├── execution_time_vs_threads.png
│   ├── speedup_vs_threads.png
│   ├── efficiency_vs_threads.png
│   ├── schedule_comparison.png
│   ├── workload_scaling.png
│   └── predicted_vs_actual.png
├── report/
│   ├── project_report.md          # Comprehensive academic project report
│   ├── viva_questions_answers.md  # 25+ Viva Voce questions & detailed answers
│   └── presentation_synopsis.md   # Presentation / PPT slide synopsis
├── app.py                # Academic Streamlit UI interface
├── Makefile              # Build automation for C programs & python workflows
└── README.md             # Project documentation and guide
```

---

## ⚙️ Prerequisites & Installation

### Requirements
- **C Compiler**: GCC with OpenMP support (`gcc-16`, `gcc-14`, or GCC 10+)
- **Python**: Version 3.10+
- **Python Libraries**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`

### Virtual Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # Or: pip install numpy pandas scikit-learn matplotlib joblib streamlit
```

---

## 🚀 Compilation & Execution Guide

### 1. Compile C Programs
```bash
make all
```
Compiles `src/sequential_pi.c`, `src/parallel_pi.c`, and `src/benchmark.c` into `bin/`.

### 2. Manual C Executions
**Sequential:**
```bash
./bin/sequential_pi 100000000
```

**Parallel:**
```bash
# Usage: ./bin/parallel_pi [N] [threads] [schedule] [chunk_size]
./bin/parallel_pi 100000000 4 static 1000
./bin/parallel_pi 100000000 8 dynamic 10000
```

### 3. Run Benchmark Suite & Generate Empirical CSV
```bash
make run-benchmark
```
Runs 216 empirical workload combinations ($N \in \{1\text{M}, 10\text{M}, 20\text{M}, 50\text{M}, 75\text{M}, 100\text{M}\}$, Threads $\in \{1, 2, 4, 8\}$, Schedules $\in \{\text{static}, \text{dynamic}, \text{guided}\}$, Chunks $\in \{100, 1000, 10000\}$) and exports results to `data/performance.csv`.

### 4. Train AI Speedup Model
```bash
make train
```
Trains a Random Forest Regressor, evaluates it on unseen workload sizes ($N = 75\text{M}$), and saves `model/speedup_model.pkl`.

### 5. Run AI-Assisted Scheduler
```bash
./venv/bin/python ai/ai_scheduler.py 80000000
```
Recommends optimal configuration for $N = 80\text{M}$, runs live C executables, and compares predicted vs actual speedup.

### 6. Generate Performance Graphs
```bash
make visualize
```
Generates 6 PNG charts in `results/`.

### 7. Launch Academic UI
```bash
make app
# Or: ./venv/bin/streamlit run app.py
```
Opens interactive Streamlit dashboard at `http://localhost:8501`.

---

## 📊 Key Formulas

1. **Midpoint Numerical Integration**:
   $$\Delta x = \frac{1}{N}, \quad x_i = (i + 0.5) \cdot \Delta x, \quad \pi \approx \Delta x \cdot \sum_{i=0}^{N-1} \frac{4}{1 + x_i^2}$$

2. **Speedup ($S$)**:
   $$S = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$$

3. **Parallel Efficiency ($E$)**:
   $$E = \frac{S}{P} \times 100\%$$
   where $P$ is the number of OpenMP threads.

---

## 📄 License & Student Declaration

Developed as an academic case study for Parallel & Distributed Computing course laboratory work.
