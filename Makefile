# Makefile for AI-Assisted OpenMP Scheduling for Parallel Numerical Computation

# Auto-detect OpenMP C compiler
GCC_16 := $(shell which /opt/homebrew/bin/gcc-16 2>/dev/null)
GCC_14 := $(shell which /opt/homebrew/bin/gcc-14 2>/dev/null)
GCC_SYS := $(shell which gcc 2>/dev/null)

ifneq ($(GCC_16),)
    CC = $(GCC_16)
    CFLAGS = -O3 -fopenmp -Wall
    LDFLAGS = -lm
else ifneq ($(GCC_14),)
    CC = $(GCC_14)
    CFLAGS = -O3 -fopenmp -Wall
    LDFLAGS = -lm
else
    CC = clang
    CFLAGS = -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -Wall
    LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -lm
endif

PYTHON = ./venv/bin/python

.PHONY: all sequential parallel benchmark run-benchmark train visualize app demo clean help

all: sequential parallel benchmark

sequential: src/sequential_pi.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -o bin/sequential_pi $(LDFLAGS)
	@echo "[BUILD SUCCESS] bin/sequential_pi"

parallel: src/parallel_pi.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -o bin/parallel_pi $(LDFLAGS)
	@echo "[BUILD SUCCESS] bin/parallel_pi"

benchmark: src/benchmark.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -o bin/benchmark $(LDFLAGS)
	@echo "[BUILD SUCCESS] bin/benchmark"

run-benchmark: benchmark
	@mkdir -p data
	./bin/benchmark

train: data/performance.csv
	$(PYTHON) ai/train_model.py

visualize: data/performance.csv
	$(PYTHON) ai/visualize.py

app:
	$(PYTHON) -m streamlit run app.py --server.headless true

demo: all
	@echo "========================================================================="
	@echo " Running Quick AI Scheduler Demonstration (N = 50,000,000 steps)..."
	@echo "========================================================================="
	$(PYTHON) ai/ai_scheduler.py 50000000

clean:
	rm -rf bin data/performance.csv model/speedup_model.pkl results/*.png
	@echo "[CLEAN COMPLETE]"

help:
	@echo "Available Makefile Targets:"
	@echo "  make all          - Compile C programs (sequential_pi, parallel_pi, benchmark)"
	@echo "  make run-benchmark- Run empirical OpenMP benchmark suite (1 warmup + 5 measured runs)"
	@echo "  make train        - Train Random Forest speedup model (GroupKFold + 75M holdout)"
	@echo "  make visualize    - Generate performance graphs"
	@echo "  make demo         - Run quick AI scheduler demonstration on N = 50M"
	@echo "  make app          - Launch Streamlit academic web interface"
	@echo "  make clean        - Remove compiled binaries and generated data"
