/*
 * sequential_pi.c
 * ------------------------------------------------------------------
 * Project: AI-Assisted Parallel Numerical Computation & Scheduling
 * Description: Calculates Pi sequentially using midpoint numerical integration.
 * Formula: Pi = integral_0^1 (4 / (1 + x^2)) dx
 * ------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// High precision timer helper
double get_time_sec(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

int main(int argc, char *argv[]) {
    long long N = 100000000LL; // Default 100M steps

    if (argc > 1) {
        N = atoll(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Error: Number of steps N must be positive.\n");
            return 1;
        }
    }

    double step = 1.0 / (double)N;
    double sum = 0.0;

    double start_time = get_time_sec();

    for (long long i = 0; i < N; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    double end_time = get_time_sec();
    double execution_time = end_time - start_time;

    double actual_pi = 3.14159265358979323846;
    double error = fabs(pi - actual_pi);

    printf("--- Sequential Pi Calculation ---\n");
    printf("Number of steps (N) : %lld\n", N);
    printf("Calculated Pi       : %.15f\n", pi);
    printf("Absolute Error      : %.15e\n", error);
    printf("Execution Time (s)  : %.6f\n", execution_time);

    return 0;
}
