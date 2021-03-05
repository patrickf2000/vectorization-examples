#include "rex_kmp.h" 
//sum.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <malloc.h>
#include <arm_sve.h> 
#define N_RUNS 10
#define N 1200
// read timer in second

double read_timer()
{
  struct timeb tm;
  ftime(&tm);
  return ((double )tm . time) + ((double )tm . millitm) / 1000.0;
}
//Create a matrix and a vector and fill with random numbers

void init(float *X)
{
  for (int i = 0; i < 120000; i++) {
    X[i] = ((float )(rand())) / ((float )(2147483647 / 10.0));
  }
}
//Our sum function- what it does is pretty straight-forward.

float sum(float *X,float *Y,float *answer)
{
  int i;
  float result = 0;
  svbool_t count = svwhilelt_b32(i, 119999);
  for (i = 0; i <= 119999; i += svcntw()) {
    svfloat32_t __vec0 = svld1(count, &X[i]);
    svfloat32_t __vec1 = svld1(count, &Y[i]);
    svfloat32_t __vec2 = svadd_f32_m(count, __vec1,__vec0);
    svst1(count, &answer[i],__vec2);
    count = svwhilelt_b32(i, 119999);
  }
  return result;
}
// Debug functions

float sum_serial(float *X,float *Y,float *answer)
{
  float result = 0;
  for (int i = 0; i < 120000; i++) {
    answer[i] = X[i] + Y[i];
  }
  return result;
}

void print_vector(float *vector)
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
  float *X = (malloc(sizeof(float ) * 120000));
  float *Y = (malloc(sizeof(float ) * 120000));
  float *answer = (malloc(sizeof(float ) * 120000));
  float *answer_serial = (malloc(sizeof(float ) * 120000));
  srand((time(((void *)0))));
  init(X);
  init(Y);
  double start = read_timer();
  for (int i = 0; i < 1000; i++) 
    sum(X,Y,answer);
  double t = read_timer() - start;
  double start_serial = read_timer();
  for (int i = 0; i < 1000; i++) 
    sum_serial(X,Y,answer_serial);
  double t_serial = read_timer() - start_serial;
  print_vector(X);
  puts("+");
  print_vector(Y);
  puts("=\n");
  printf("SIMD:\n");
  print_vector(answer);
  puts("---------------------------------");
  printf("Serial:\n");
  print_vector(answer_serial);
  double gflops = 2.0 * 120000 * 120000 * 1000 / (1.0e9 * t);
  double gflops_serial = 2.0 * 120000 * 120000 * 1000 / (1.0e9 * t_serial);
  printf("==================================================================\n");
  printf("Performance:\t\t\tRuntime (s)\t GFLOPS\n");
  printf("------------------------------------------------------------------\n");
  printf("Sum (SIMD):\t\t%4f\t%4f\n",t,gflops);
  printf("Sum (Serial):\t\t%4f\t%4f\n",t_serial,gflops_serial);
  free(X);
  free(Y);
  free(answer);
  free(answer_serial);
  return 0;
}
