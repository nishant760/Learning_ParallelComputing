/*
 * benchmark.c
 * ------------------------------------------------------------------
 * Project: AI-Assisted Parallel Numerical Computation & Scheduling
 * Description: Automated benchmark suite that runs sequential and parallel
 *              Pi computations across multiple workloads, thread counts,
 *              schedules, and chunk sizes. Outputs empirical metrics into data/performance.csv.
 * ------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

volatile double g_dummy_sink = 0.0;

// Measure sequential execution time for a given N
__attribute__((noinline))
double run_sequential(long long N, double *out_pi) {
    double step = 1.0 / (double)N;
    double sum = 0.0;
    
    double start_time = omp_get_wtime();
    __asm__ __volatile__("" ::: "memory");

    for (long long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    __asm__ __volatile__("" ::: "memory");
    double end_time = omp_get_wtime();
    
    *out_pi = step * sum;
    g_dummy_sink = *out_pi; // Ensure result cannot be eliminated by GCC
    return end_time - start_time;
}

// Measure parallel execution time for a given configuration
__attribute__((noinline))
double run_parallel(long long N, int num_threads, const char *sched_type, int chunk_size, double *out_pi) {
    omp_set_num_threads(num_threads);

    if (strcmp(sched_type, "static") == 0) {
        omp_set_schedule(omp_sched_static, chunk_size);
    } else if (strcmp(sched_type, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, chunk_size);
    } else if (strcmp(sched_type, "guided") == 0) {
        omp_set_schedule(omp_sched_guided, chunk_size);
    } else {
        omp_set_schedule(omp_sched_static, chunk_size);
    }

    double step = 1.0 / (double)N;
    double sum = 0.0;
    
    double start_time = omp_get_wtime();
    __asm__ __volatile__("" ::: "memory");

    #pragma omp parallel for reduction(+:sum) schedule(runtime)
    for (long long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    __asm__ __volatile__("" ::: "memory");
    double end_time = omp_get_wtime();
    
    *out_pi = step * sum;
    g_dummy_sink = *out_pi;
    return end_time - start_time;
}

int main(void) {
    long long input_sizes[] = {1000000LL, 10000000LL, 20000000LL, 50000000LL, 75000000LL, 100000000LL};
    int num_sizes = sizeof(input_sizes) / sizeof(input_sizes[0]);

    int thread_counts[] = {1, 2, 4, 8};
    int num_threads_arr = sizeof(thread_counts) / sizeof(thread_counts[0]);

    const char *schedules[] = {"static", "dynamic", "guided"};
    int num_schedules = sizeof(schedules) / sizeof(schedules[0]);

    int chunk_sizes[] = {100, 1000, 10000};
    int num_chunks = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    const char *output_file = "data/performance.csv";
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        perror("Failed to open output CSV file");
        return 1;
    }

    fprintf(fp, "N,threads,schedule,chunk,execution_time,speedup,efficiency\n");
    printf("=================================================================================\n");
    printf(" Starting OpenMP Parallel Pi Computation Benchmark Suite\n");
    printf("=================================================================================\n");

    int total_runs = num_sizes * num_threads_arr * num_schedules * num_chunks;
    int current_run = 0;

    for (int s = 0; s < num_sizes; s++) {
        long long N = input_sizes[s];

        // Baseline sequential execution time (min of 3 runs)
        double seq_time = 1e9;
        double pi_val = 0.0;
        for (int r = 0; r < 3; r++) {
            double dummy_pi = 0.0;
            double t = run_sequential(N, &dummy_pi);
            if (t < seq_time) {
                seq_time = t;
                pi_val = dummy_pi;
            }
        }
        g_dummy_sink += pi_val;
        printf("\n>>> Workload N = %lld | Baseline Sequential Time: %.6f s | Pi: %.10f <<<\n", N, seq_time, pi_val);

        for (int t_idx = 0; t_idx < num_threads_arr; t_idx++) {
            int threads = thread_counts[t_idx];

            for (int sched_idx = 0; sched_idx < num_schedules; sched_idx++) {
                const char *sched = schedules[sched_idx];

                for (int c_idx = 0; c_idx < num_chunks; c_idx++) {
                    int chunk = chunk_sizes[c_idx];
                    current_run++;

                    double par_time = 1e9;
                    for (int r = 0; r < 3; r++) {
                        double dummy_pi = 0.0;
                        double t = run_parallel(N, threads, sched, chunk, &dummy_pi);
                        if (t < par_time) par_time = t;
                    }

                    double speedup = seq_time / par_time;
                    double efficiency = (speedup / (double)threads) * 100.0;

                    fprintf(fp, "%lld,%d,%s,%d,%.8f,%.4f,%.2f\n",
                            N, threads, sched, chunk, par_time, speedup, efficiency);
                    fflush(fp);

                    printf("[%3d/%3d] N=%10lld | T=%d | Sched=%-7s | Chunk=%-5d -> Time: %8.6fs | Speedup: %5.2fx | Eff: %6.2f%%\n",
                           current_run, total_runs, N, threads, sched, chunk, par_time, speedup, efficiency);
                }
            }
        }
    }

    fclose(fp);
    printf("=================================================================================\n");
    printf(" Benchmarking complete! Empirical dataset written to: %s\n", output_file);
    printf(" Total sink accumulation check: %.10f\n", g_dummy_sink);
    printf("=================================================================================\n");

    return 0;
}
