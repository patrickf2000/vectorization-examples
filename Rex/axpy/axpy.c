//axpy.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <malloc.h>

#define N_RUNS 1000
#define N 1200

// read timer in second
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

//Create a matrix and a vector and fill with random numbers
void init(float *X, float *Y) {
    for (int i = 0; i<N; i++) {
        X[i] = (float)rand()/(float)(RAND_MAX/10.0);
        Y[i] = (float)rand()/(float)(RAND_MAX/10.0);
    }
}

//Our sum function- what it does is pretty straight-forward.
void axpy(float *X, float *Y, float a) {
    #pragma omp simd
    for (int i = 0; i<N; i++) {
        Y[i] += a * X[i];
    }
}

// Debug functions
void axpy_serial(float *X, float *Y, float a) {
    for (int i = 0; i<N; i++) {
        Y[i] += a * X[i];
    }
}

void print_vector(float *vector) {
    printf("[");
    for (int i = 0; i<8; i++) {
        printf("%.2f ", vector[i]);
    }
    puts("]");
}

float check(float *A, float *B){
    float difference = 0;
    for(int i = 0;i<N; i++){
        difference += A[i]- B[i];
    }
    return difference;
}

int main(int argc, char **argv) {
    //Set everything up
    float *X = malloc(sizeof(float)*N);
    float *Y = malloc(sizeof(float)*N);
    float *Y_serial = malloc(sizeof(float)*N);
    float a = 3.14;
    
    srand(time(NULL));
    init(X, Y);
    for (int i = 0; i<N; i++) Y_serial[i] = Y[i];
    
    print_vector(Y);
    print_vector(X);
    printf("%.2f\n", a);
    puts("=\n");
    
    double start = read_timer();
    for (int i = 0; i<N_RUNS; i++)
        axpy(X, Y, a);
    double t = (read_timer() - start);
    
    double start_serial = read_timer();
    for (int i = 0; i<N_RUNS; i++)
        axpy_serial(X, Y_serial, a);
    double t_serial = (read_timer() - start_serial);
    
    print_vector(Y);
    puts("---------------------------------");
    print_vector(Y_serial);
    
    double gflops = ((2.0 * N) * N * N_RUNS) / (1.0e9 * t);
    double gflops_serial = ((2.0 * N) * N * N_RUNS) / (1.0e9 * t_serial);
    
    printf("==================================================================\n");
    printf("Performance:\t\t\tRuntime (s)\t GFLOPS\n");
    printf("------------------------------------------------------------------\n");
    printf("AXPY (SIMD):\t\t%4f\t%4f\n", t, gflops);
    printf("AXPY (Serial):\t\t%4f\t%4f\n", t_serial, gflops_serial);
    printf("Correctness check: %f\n", check(Y,Y_serial));
    
    free(X);
    free(Y);
    free(Y_serial);
    
    return 0;    
}

