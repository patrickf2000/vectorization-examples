//sum.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <malloc.h>

#define N_RUNS 1000
#define N 120000

// read timer in second
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

//Create a matrix and a vector and fill with random numbers
void init(int *X) {
    for (int i = 0; i<N; i++) {
        X[i] = (int)rand()/(int)(RAND_MAX/10.0);
    }
}

//Our sum function- what it does is pretty straight-forward.
int sum(int *X) {
    int result = 0;
    
    #pragma omp simd reduction(+:result)
    for (int i = 0; i<N; i++) {
        result += X[i];
    }
    
    return result;
}

// Debug functions
int sum_serial(int *X) {
    int result = 0;
    
    for (int i = 0; i<N; i++) {
        result += X[i];
    }
    
    return result;
}

void print_vector(int *vector) {
    printf("[");
    for (int i = 0; i<8; i++) {
        printf("%d ", vector[i]);
    }
    puts("]");
}

int main(int argc, char **argv) {
    //Set everything up
    int *X = malloc(sizeof(int)*N);
    int result, result_serial;
    
    srand(time(NULL));
    init(X);
    
    double start = read_timer();
    for (int i = 0; i<N_RUNS; i++)
        result = sum(X);
    double t = (read_timer() - start);
    
    double start_serial = read_timer();
    for (int i = 0; i<N_RUNS; i++)
        result_serial = sum_serial(X);
    double t_serial = (read_timer() - start_serial);
    
    print_vector(X);
    puts("=\n");
    printf("SIMD: %d\n", result);
    puts("---------------------------------");
    printf("Serial: %d\n", result_serial);
    
    double gflops = ((2.0 * N) * N * N_RUNS) / (1.0e9 * t);
    double gflops_serial = ((2.0 * N) * N * N_RUNS) / (1.0e9 * t_serial);
    
    printf("==================================================================\n");
    printf("Performance:\t\t\tRuntime (s)\t GFLOPS\n");
    printf("------------------------------------------------------------------\n");
    printf("Sum (SIMD):\t\t%4f\t%4f\n", t, gflops);
    printf("Sum (Serial):\t\t%4f\t%4f\n", t_serial, gflops_serial);
    printf("Correctness check: %d\n", result_serial - result);
    
    free(X);
    
    return 0;    
}

