#include "rex_kmp.h" 
//axpy.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <malloc.h>
#include <immintrin.h> 
#define N_RUNS 1000
#define N 1200
// read timer in second

double read_timer()
{
  struct timeb tm;
  ftime(&tm);
  return ((double )tm . time) + ((double )tm . millitm) / 1000.0;
}
//Create a matrix and a vector and fill with random numbers

void init(int *X,int *Y)
{
  for (int i = 0; i < 1200; i++) {
    X[i] = ((int )(rand())) / ((int )(2147483647 / 10.0));
    Y[i] = ((int )(rand())) / ((int )(2147483647 / 10.0));
  }
}
//Our sum function- what it does is pretty straight-forward.

void axpy(int *X,int *Y,int a)
{
  int i;
  __m512i __vec1 = _mm512_set1_epi32(a);
  for (i = 0; i <= 1199; i += 16) {
    __m512i __vec0 = _mm512_loadu_si512((__m512i *)(&Y[i]));
    __m512i __vec2 = _mm512_loadu_si512((__m512i *)(&X[i]));
    __m512i __vec3 = _mm512_mullo_epi32(__vec2,__vec1);
    __m512i __vec4 = _mm512_add_epi32(__vec3,__vec0);
    _mm512_storeu_si512(&Y[i],__vec4);
  }
}
// Debug functions

void axpy_serial(int *X,int *Y,int a)
{
  for (int i = 0; i < 1200; i++) {
    Y[i] += a * X[i];
  }
}

void print_vector(int *vector)
{
  printf("[");
  for (int i = 0; i < 8; i++) {
    printf("%d ",vector[i]);
  }
  puts("]");
}

int check(int *A,int *B)
{
  int difference = 0;
  for (int i = 0; i < 1200; i++) {
    difference += A[i] - B[i];
  }
  return difference;
}

int main(int argc,char **argv)
{
  int status = 0;
//Set everything up
  int *X = (malloc(sizeof(int ) * 1200));
  int *Y = (malloc(sizeof(int ) * 1200));
  int *Y_serial = (malloc(sizeof(int ) * 1200));
  int a = 3;
  srand((time(((void *)0))));
  init(X,Y);
  for (int i = 0; i < 1200; i++) 
    Y_serial[i] = Y[i];
  print_vector(Y);
  print_vector(X);
  printf("%d\n",a);
  puts("=\n");
  double start = read_timer();
  for (int i = 0; i < 1000; i++) 
    axpy(X,Y,a);
  double t = read_timer() - start;
  double start_serial = read_timer();
  for (int i = 0; i < 1000; i++) 
    axpy_serial(X,Y_serial,a);
  double t_serial = read_timer() - start_serial;
  print_vector(Y);
  puts("---------------------------------");
  print_vector(Y_serial);
  double gflops = 2.0 * 1200 * 1200 * 1000 / (1.0e9 * t);
  double gflops_serial = 2.0 * 1200 * 1200 * 1000 / (1.0e9 * t_serial);
  printf("==================================================================\n");
  printf("Performance:\t\t\tRuntime (s)\t GFLOPS\n");
  printf("------------------------------------------------------------------\n");
  printf("AXPY (SIMD):\t\t%4f\t%4f\n",t,gflops);
  printf("AXPY (Serial):\t\t%4f\t%4f\n",t_serial,gflops_serial);
  printf("Correctness check: %d\n",(check(Y,Y_serial)));
  free(X);
  free(Y);
  free(Y_serial);
  return 0;
}
