# AI-Assisted OpenMP Scheduling for Parallel Numerical Computation

A student-developed parallel computing project demonstrating workload-aware OpenMP performance tuning using C, OpenMP multi-threading, and Machine Learning (Random Forest Regression).

> **Note**: Midpoint numerical integration of $\pi = \int_0^1 \frac{4}{1+x^2} dx$ is the computational workload used to study OpenMP performance tuning.

---

## 📌 Project Overview & Workflow

1. **Input**: Workload iteration count $N$.
2. **Parameters**: Thread count (2, 4, 8), Schedule strategy (`static`, `dynamic`, `guided`), Chunk size (100, 1000, 10000).
3. **Benchmarking**: Execute C/OpenMP numerical routines (1 warm-up run + 5 measured runs per config, taking median time).
4. **Dataset**: Empirical performance stored in `data/performance.csv`.
5. **Machine Learning**: Train Random Forest Regressor to predict speedup $(N, \text{threads}, \text{schedule}, \text{chunk}) \to \text{speedup}$.
6. **AI Scheduler**: Rank 27 candidate configurations for target $N$ and recommend optimal settings.
7. **Validation**: Execute recommended C/OpenMP configuration live and compare **Predicted** vs **Measured** speedup.

---

## 🏗️ Architecture Flow

```text
User Workload N ──> Candidate Configs ──> Random Forest Model ──> Top Recommendation
                                                                         │
                                                                         v
Predicted vs Measured Speedup Validation <── Speedup Metrics <── Live C/OpenMP Execution
```

---

## 📁 Repository Structure

```
ParallelComputing/
├── src/
│   ├── sequential_pi.c          # Single-threaded baseline C implementation (T1)
│   ├── parallel_pi.c            # Multi-threaded OpenMP C implementation with reduction
│   └── benchmark.c              # Empirical benchmarking suite (warmup + median timing)
├── ai/
│   ├── train_model.py           # Random Forest Regressor (GroupKFold + N=75M holdout)
│   ├── ai_scheduler.py          # AI scheduler & live execution validator
│   ├── compare_schedules.py     # Live comparison tool for all OpenMP schedules
│   └── visualize.py             # Matplotlib graph generator
├── data/
│   └── performance.csv          # Empirical benchmark dataset (median times, speedups, etc.)
├── model/
│   └── speedup_model.pkl        # Saved Scikit-Learn Random Forest pipeline
├── results/                     # Performance plot charts
├── report/
│   ├── project_report.md         # Academic project report
│   ├── viva_questions_answers.md # Viva Voce Q&As
│   ├── presentation_synopsis.md  # Presentation synopsis
│   └── demo_guide.md            # Step-by-step professor demonstration guide
├── app.py                       # Streamlit UI dashboard
├── Makefile                     # Build & run automation script
├── requirements.txt             # Minimal Python dependencies
└── README.md                    # Project guide
```

---

## ⚙️ Installation & Commands

```bash
# 1. Environment & Dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Compile C binaries
make all

# 3. Run empirical benchmark suite (1 warmup + 5 measured runs per config)
make run-benchmark

# 4. Train Random Forest model (GroupKFold CV + N=75M holdout)
make train

# 5. Run quick AI scheduler demo (N = 50M)
make demo

# 6. Run AI scheduler for custom workload (e.g., N = 80M)
./venv/bin/python ai/ai_scheduler.py 80000000

# 7. Generate performance graphs
make visualize

# 8. Launch Streamlit web dashboard
make app
```

---

## 📊 Key Formulas

- **Midpoint Rule**: $h = \frac{1}{N}, \ x_i = (i + 0.5)h, \ \pi \approx h \sum_{i=0}^{N-1} \frac{4}{1 + x_i^2}$
- **Speedup**: $S = \frac{T_{1,\text{median}}}{T_{p,\text{median}}}$
- **Efficiency**: $E = \frac{S}{P} \times 100\%$

---

## 📄 Academic Note

> ⚠️ **Hardware Dependence Note**: Performance metrics and speedups are hardware-dependent for the current host system.
