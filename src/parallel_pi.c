/*
 * parallel_pi.c
 * ------------------------------------------------------------------
 * Project: AI-Assisted Parallel Numerical Computation & Scheduling
 * Description: Calculates Pi in parallel using OpenMP midpoint integration.
 * Features:
 *   - Configurable thread counts (omp_set_num_threads)
 *   - OpenMP reduction clause (reduction(+:sum))
 *   - Configurable scheduling strategies: static, dynamic, guided
 *   - Configurable chunk size
 * ------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    long long N = 100000000LL; // Default 100M
    int num_threads = 4;
    char sched_type[32] = "static";
    int chunk_size = 1000;

    if (argc > 1) N = atoll(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) strncpy(sched_type, argv[3], sizeof(sched_type) - 1);
    if (argc > 4) chunk_size = atoi(argv[4]);

    if (N <= 0 || num_threads <= 0 || chunk_size <= 0) {
        fprintf(stderr, "Invalid parameters. Usage: %s [N] [threads] [schedule] [chunk_size]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(num_threads);

    // Set schedule type using runtime configuration
    if (strcmp(sched_type, "static") == 0) {
        omp_set_schedule(omp_sched_static, chunk_size);
    } else if (strcmp(sched_type, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, chunk_size);
    } else if (strcmp(sched_type, "guided") == 0) {
        omp_set_schedule(omp_sched_guided, chunk_size);
    } else {
        fprintf(stderr, "Unknown schedule '%s'. Defaulting to static.\n", sched_type);
        omp_set_schedule(omp_sched_static, chunk_size);
        strcpy(sched_type, "static");
    }

    double step = 1.0 / (double)N;
    double sum = 0.0;

    double start_time = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum) schedule(runtime)
    for (long long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    double actual_pi = 3.14159265358979323846;
    double error = fabs(pi - actual_pi);

    printf("--- Parallel Pi Calculation (OpenMP) ---\n");
    printf("Number of steps (N) : %lld\n", N);
    printf("Threads             : %d\n", num_threads);
    printf("Schedule            : %s\n", sched_type);
    printf("Chunk Size          : %d\n", chunk_size);
    printf("Calculated Pi       : %.15f\n", pi);
    printf("Absolute Error      : %.15e\n", error);
    printf("Execution Time (s)  : %.6f\n", execution_time);

    return 0;
}
