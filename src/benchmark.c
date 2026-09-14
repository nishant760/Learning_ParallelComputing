/*
 * benchmark.c
 * ------------------------------------------------------------------
 * Project: AI-Assisted Parallel Numerical Computation & Scheduling
 * Description: Automated benchmark suite that runs true sequential and
 *              OpenMP parallel Pi computations across multiple workloads,
 *              thread counts, schedules, and chunk sizes.
 *
 * Methodology (Phases 1-3):
 *   - Performs 1 warm-up run + 5 measured runs per configuration.
 *   - Computes median, mean, standard deviation, and min execution times.
 *   - Uses MEDIAN execution time as primary metric for Speedup & Efficiency.
 *   - Compares true sequential execution against OpenMP parallel execution.
 *   - Exports dataset into data/performance.csv.
 * ------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define WARMUP_RUNS 1
#define MEASURED_RUNS 5

volatile double g_dummy_sink = 0.0;

// Measure single-threaded sequential execution time for a given N (conceptual T1 baseline)
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
    g_dummy_sink = *out_pi; // Prevent dead-code elimination by compiler
    return end_time - start_time;
}

// Measure multi-threaded OpenMP parallel execution time for a given configuration
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

int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

typedef struct {
    double median;
    double mean;
    double stddev;
    double min;
    double pi;
} RunStats;

RunStats measure_sequential_stats(long long N) {
    double dummy_pi = 0.0;

    // 1. Warm-up run
    for (int w = 0; w < WARMUP_RUNS; w++) {
        run_sequential(N, &dummy_pi);
    }

    // 2. Measured runs
    double times[MEASURED_RUNS];
    double pi_vals[MEASURED_RUNS];
    double sum_t = 0.0;

    for (int r = 0; r < MEASURED_RUNS; r++) {
        times[r] = run_sequential(N, &pi_vals[r]);
        sum_t += times[r];
    }

    double mean_t = sum_t / (double)MEASURED_RUNS;

    double var_t = 0.0;
    for (int r = 0; r < MEASURED_RUNS; r++) {
        double diff = times[r] - mean_t;
        var_t += diff * diff;
    }
    double stddev_t = sqrt(var_t / (double)MEASURED_RUNS);

    double sorted_times[MEASURED_RUNS];
    memcpy(sorted_times, times, sizeof(times));
    qsort(sorted_times, MEASURED_RUNS, sizeof(double), compare_doubles);

    RunStats stats;
    stats.median = sorted_times[MEASURED_RUNS / 2];
    stats.min = sorted_times[0];
    stats.mean = mean_t;
    stats.stddev = stddev_t;
    stats.pi = pi_vals[0];
    return stats;
}

RunStats measure_parallel_stats(long long N, int threads, const char *sched, int chunk) {
    double dummy_pi = 0.0;

    // 1. Warm-up run
    for (int w = 0; w < WARMUP_RUNS; w++) {
        run_parallel(N, threads, sched, chunk, &dummy_pi);
    }

    // 2. Measured runs
    double times[MEASURED_RUNS];
    double pi_vals[MEASURED_RUNS];
    double sum_t = 0.0;

    for (int r = 0; r < MEASURED_RUNS; r++) {
        times[r] = run_parallel(N, threads, sched, chunk, &pi_vals[r]);
        sum_t += times[r];
    }

    double mean_t = sum_t / (double)MEASURED_RUNS;

    double var_t = 0.0;
    for (int r = 0; r < MEASURED_RUNS; r++) {
        double diff = times[r] - mean_t;
        var_t += diff * diff;
    }
    double stddev_t = sqrt(var_t / (double)MEASURED_RUNS);

    double sorted_times[MEASURED_RUNS];
    memcpy(sorted_times, times, sizeof(times));
    qsort(sorted_times, MEASURED_RUNS, sizeof(double), compare_doubles);

    RunStats stats;
    stats.median = sorted_times[MEASURED_RUNS / 2];
    stats.min = sorted_times[0];
    stats.mean = mean_t;
    stats.stddev = stddev_t;
    stats.pi = pi_vals[0];
    return stats;
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

    // New revised dataset header (Phase 2)
    fprintf(fp, "N,threads,schedule,chunk,median_time,mean_time,stddev_time,min_time,speedup,efficiency,pi_error\n");
    printf("=================================================================================\n");
    printf(" Starting OpenMP Empirical Benchmark Suite (Warmup: %d, Measured Runs: %d)\n", WARMUP_RUNS, MEASURED_RUNS);
    printf("=================================================================================\n");

    int total_configs = num_sizes * num_threads_arr * num_schedules * num_chunks;
    int current_run = 0;
    const double exact_pi = 3.14159265358979323846;

    for (int s = 0; s < num_sizes; s++) {
        long long N = input_sizes[s];

        // Measure true sequential baseline stats (1 warm-up + 5 measured runs)
        RunStats seq_stats = measure_sequential_stats(N);
        g_dummy_sink += seq_stats.pi;

        printf("\n>>> Workload N = %10lld | Sequential Median Time: %.6f s (StdDev: %.6f) | Pi: %.10f <<<\n",
               N, seq_stats.median, seq_stats.stddev, seq_stats.pi);

        for (int t_idx = 0; t_idx < num_threads_arr; t_idx++) {
            int threads = thread_counts[t_idx];

            for (int sched_idx = 0; sched_idx < num_schedules; sched_idx++) {
                const char *sched = schedules[sched_idx];

                for (int c_idx = 0; c_idx < num_chunks; c_idx++) {
                    int chunk = chunk_sizes[c_idx];
                    current_run++;

                    RunStats par_stats = measure_parallel_stats(N, threads, sched, chunk);
                    g_dummy_sink += par_stats.pi;

                    // Speedup & Efficiency calculated using MEDIAN time
                    double speedup = seq_stats.median / par_stats.median;
                    double efficiency = (speedup / (double)threads) * 100.0;
                    double pi_error = fabs(par_stats.pi - exact_pi);

                    fprintf(fp, "%lld,%d,%s,%d,%.8f,%.8f,%.8f,%.8f,%.4f,%.2f,%.15e\n",
                            N, threads, sched, chunk,
                            par_stats.median, par_stats.mean, par_stats.stddev, par_stats.min,
                            speedup, efficiency, pi_error);
                    fflush(fp);

                    printf("[%3d/%3d] N=%10lld | T=%d | Sched=%-7s | Chunk=%-5d -> Median: %8.6fs | Speedup: %5.2fx | Eff: %6.2f%%\n",
                           current_run, total_configs, N, threads, sched, chunk, par_stats.median, speedup, efficiency);
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
