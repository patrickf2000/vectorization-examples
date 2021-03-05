#include "rex_kmp.h" 
//sum.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <malloc.h>
#include <immintrin.h> 
#define N_RUNS 1000
#define N 120000
// read timer in second

double read_timer()
{
  struct timeb tm;
  ftime(&tm);
  return ((double )tm . time) + ((double )tm . millitm) / 1000.0;
}
//Create a matrix and a vector and fill with random numbers

void init(double *X)
{
  for (int i = 0; i < 120000; i++) {
    X[i] = ((double )(rand())) / ((double )(2147483647 / 10.0));
  }
}
//Our sum function- what it does is pretty straight-forward.

double sum(double *X)
{
  int i;
  double result = 0;
  __m512d __vec0 = _mm512_set1_pd(result);
  __m512d __part3 = _mm512_setzero_pd();
  for (i = 0; i <= 119999; i += 8) {
    __m512d __vec1 = _mm512_loadu_pd(&X[i]);
    __m512d __vec2 = _mm512_add_pd(__vec1,__vec0);
    __part3 = _mm512_add_pd(__part3,__vec2);
  }
  __m256d __buf0 = _mm512_extractf64x4_pd(__part3,0);
  __m256d __buf1 = _mm512_extractf64x4_pd(__part3,1);
  __buf1 = _mm256_add_pd(__buf0,__buf1);
  __buf1 = _mm256_hadd_pd(__buf1,__buf1);
  double __buf2[4];
  _mm256_storeu_pd(&__buf2,__buf1);
  result = __buf2[0] + __buf2[2];
  return result;
}
// Debug functions

double sum_serial(double *X)
{
  double result = 0;
  for (int i = 0; i < 120000; i++) {
    result += X[i];
  }
  return result;
}

void print_vector(double *vector)
{
  printf("[");
  for (int i = 0; i < 8; i++) {
    printf("%.2f ",vector[i]);
  }
  puts("]");
}

int main(int argc,char **argv)
{
  int status = 0;
//Set everything up
  double *X = (malloc(sizeof(double ) * 120000));
  double result;
  double result_serial;
  srand((time(((void *)0))));
  init(X);
  double start = read_timer();
  for (int i = 0; i < 1000; i++) 
    result = sum(X);
  double t = read_timer() - start;
  double start_serial = read_timer();
  for (int i = 0; i < 1000; i++) 
    result_serial = sum_serial(X);
  double t_serial = read_timer() - start_serial;
  print_vector(X);
  puts("=\n");
  printf("SIMD: %f\n",result);
  puts("---------------------------------");
  printf("Serial: %f\n",result_serial);
  double gflops = 2.0 * 120000 * 120000 * 1000 / (1.0e9 * t);
  double gflops_serial = 2.0 * 120000 * 120000 * 1000 / (1.0e9 * t_serial);
  printf("==================================================================\n");
  printf("Performance:\t\t\tRuntime (s)\t GFLOPS\n");
  printf("------------------------------------------------------------------\n");
  printf("Sum (SIMD):\t\t%4f\t%4f\n",t,gflops);
  printf("Sum (Serial):\t\t%4f\t%4f\n",t_serial,gflops_serial);
  printf("Correctness check: %f\n",result_serial - result);
  free(X);
  return 0;
}
