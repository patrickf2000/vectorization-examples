#include "rex_kmp.h" 
/*
 * Square matrix multiplication
 * A[N][N] * B[N][N] = C[N][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <malloc.h>
#include <immintrin.h> 
#define N 512
//#define N 16
// read timer in second

double read_timer()
{
  struct timeb tm;
  ftime(&tm);
  return ((double )tm . time) + ((double )tm . millitm) / 1000.0;
}

void init(int **A)
{
  int i;
  int j;
  for (i = 0; i < 512; i++) {
    for (j = 0; j < 512; j++) {
      A[i][j] = ((int )(rand())) / ((int )(2147483647 / 10.0));
    }
  }
}

void matmul_simd(int **A,int **B,int **C)
{
  int i;
  int j;
  int k;
  int temp;
  for (i = 0; i < 512; i++) {
    for (j = 0; j < 512; j++) {
      temp = 0;
      __m512i __vec0 = _mm512_set1_epi32(temp);
      __m512i __part7 = _mm512_setzero_epi32();
      for (k = 0; k <= 511; k += 16) {
        int *__ptr1 = A[i];
        __m512i __vec2 = _mm512_loadu_si512((__m512i *)(&__ptr1[k]));
        int *__ptr3 = B[j];
        __m512i __vec4 = _mm512_loadu_si512((__m512i *)(&__ptr3[k]));
        __m512i __vec5 = _mm512_mullo_epi32(__vec4,__vec2);
        __m512i __vec6 = _mm512_add_epi32(__vec5,__vec0);
        __part7 = _mm512_add_epi32(__part7,__vec6);
      }
      __m256i __buf0 = _mm512_extracti32x8_epi32(__part7,0);
      __m256i __buf1 = _mm512_extracti32x8_epi32(__part7,1);
      __buf1 = _mm256_add_epi32(__buf0,__buf1);
      __buf1 = _mm256_hadd_epi32(__buf1,__buf1);
      __buf1 = _mm256_hadd_epi32(__buf1,__buf1);
      int __buf2[8];
      _mm256_storeu_si256((__m256i *)(&__buf2),__buf1);
      temp = __buf2[0] + __buf2[6];
      C[i][j] = temp;
    }
  }
}
// Debug functions

void print_matrix(int **matrix)
{
  for (int i = 0; i < 8; i++) {
    printf("[");
    for (int j = 0; j < 8; j++) {
      printf("%d ",matrix[i][j]);
    }
    puts("]");
  }
  puts("");
}

void matmul_serial(int **A,int **B,int **C)
{
  int i;
  int j;
  int k;
  int temp;
  for (i = 0; i < 512; i++) {
    for (j = 0; j < 512; j++) {
      temp = 0;
      for (k = 0; k < 512; k++) {
        temp += A[i][k] * B[j][k];
      }
      C[i][j] = temp;
    }
  }
}

int check(int **A,int **B)
{
  int difference = 0;
  for (int i = 0; i < 512; i++) {
    for (int j = 0; j < 512; j++) {
      difference += A[i][j] - B[i][j];
    }
  }
  return difference;
}
// Main

int main(int argc,char *argv[])
{
  int status = 0;
//Set everything up
  int **A = (malloc(sizeof(int *) * 512));
  int **B = (malloc(sizeof(int *) * 512));
  int **C_simd = (malloc(sizeof(int *) * 512));
  int **C_serial = (malloc(sizeof(int *) * 512));
  int **BT = (malloc(sizeof(int *) * 512));
  for (int i = 0; i < 512; i++) {
    A[i] = (malloc(sizeof(int ) * 512));
    B[i] = (malloc(sizeof(int ) * 512));
    C_simd[i] = (malloc(sizeof(int ) * 512));
    C_serial[i] = (malloc(sizeof(int ) * 512));
    BT[i] = (malloc(sizeof(int ) * 512));
  }
  srand((time(((void *)0))));
  init(A);
  init(B);
  for (int line = 0; line < 512; line++) {
    for (int col = 0; col < 512; col++) {
      BT[line][col] = B[col][line];
    }
  }
  int i;
  int num_runs = 10;
  double elapsed = read_timer();
  for (i = 0; i < num_runs; i++) 
    matmul_simd(A,BT,C_simd);
  elapsed = read_timer() - elapsed;
  double elapsed_serial = read_timer();
  for (i = 0; i < num_runs; i++) 
    matmul_serial(A,BT,C_serial);
  elapsed_serial = read_timer() - elapsed_serial;
  print_matrix(A);
  print_matrix(BT);
  puts("=\n");
  print_matrix(C_simd);
  puts("---------------------------------");
  print_matrix(C_serial);
  double gflops_omp = 2.0 * 512 * 512 * 512 * num_runs / (1.0e9 * elapsed);
  double gflops_serial = 2.0 * 512 * 512 * 512 * num_runs / (1.0e9 * elapsed_serial);
  printf("======================================================================================================\n");
  printf("\tMatrix Multiplication: A[N][N] * B[N][N] = C[N][N], N=%d\n",512);
  printf("------------------------------------------------------------------------------------------------------\n");
  printf("Performance:\t\tRuntime (s)\t GFLOPS\n");
  printf("------------------------------------------------------------------------------------------------------\n");
  printf("matmul_omp:\t\t%4f\t%4f\n",elapsed,gflops_omp);
  printf("matmul_serial:\t\t%4f\t%4f\n",elapsed_serial,gflops_serial);
  printf("Correctness check: %d\n",(check(C_simd,C_serial)));
  return 0;
}
