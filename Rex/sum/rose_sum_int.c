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

void init(int *X)
{
  for (int i = 0; i < 120000; i++) {
    X[i] = ((int )(rand())) / ((int )(2147483647 / 10.0));
  }
}
//Our sum function- what it does is pretty straight-forward.

int sum(int *X)
{
  int i;
  int result = 0;
  __m512i __vec0 = _mm512_set1_epi32(result);
  __m512i __part3 = _mm512_setzero_epi32();
  for (i = 0; i <= 119999; i += 16) {
    __m512i __vec1 = _mm512_loadu_si512((__m512i *)(&X[i]));
    __m512i __vec2 = _mm512_add_epi32(__vec1,__vec0);
    __part3 = _mm512_add_epi32(__part3,__vec2);
  }
  __m256i __buf0 = _mm512_extracti32x8_epi32(__part3,0);
  __m256i __buf1 = _mm512_extracti32x8_epi32(__part3,1);
  __buf1 = _mm256_add_epi32(__buf0,__buf1);
  __buf1 = _mm256_hadd_epi32(__buf1,__buf1);
  __buf1 = _mm256_hadd_epi32(__buf1,__buf1);
  int __buf2[8];
  _mm256_storeu_si256((__m256i *)(&__buf2),__buf1);
  result = __buf2[0] + __buf2[6];
  return result;
}
// Debug functions

int sum_serial(int *X)
{
  int result = 0;
  for (int i = 0; i < 120000; i++) {
    result += X[i];
  }
  return result;
}

void print_vector(int *vector)
{
  printf("[");
  for (int i = 0; i < 8; i++) {
    printf("%d ",vector[i]);
  }
  puts("]");
}

int main(int argc,char **argv)
{
  int status = 0;
//Set everything up
  int *X = (malloc(sizeof(int ) * 120000));
  int result;
  int result_serial;
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
  printf("SIMD: %d\n",result);
  puts("---------------------------------");
  printf("Serial: %d\n",result_serial);
  double gflops = 2.0 * 120000 * 120000 * 1000 / (1.0e9 * t);
  double gflops_serial = 2.0 * 120000 * 120000 * 1000 / (1.0e9 * t_serial);
  printf("==================================================================\n");
  printf("Performance:\t\t\tRuntime (s)\t GFLOPS\n");
  printf("------------------------------------------------------------------\n");
  printf("Sum (SIMD):\t\t%4f\t%4f\n",t,gflops);
  printf("Sum (Serial):\t\t%4f\t%4f\n",t_serial,gflops_serial);
  printf("Correctness check: %d\n",result_serial - result);
  free(X);
  return 0;
}
