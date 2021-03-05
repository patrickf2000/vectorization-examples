#include "rex_kmp.h" 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <malloc.h>
#include <immintrin.h> 
#include <immintrin.h> 
#define REAL float

static double read_timer_ms()
{
  struct timeb tm;
  ftime(&tm);
  return ((double )tm . time) * 1000.0 + ((double )tm . millitm);
}
/************************************************************
 * program to solve a finite difference
 * discretization of Helmholtz equation :
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * using Jacobi iterative method.
 *
 * Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
 *
 * This c version program is translated by
 * Chunhua Liao, University of Houston, Jan, 2005
 *
 * Directives are used in this code to achieve parallelism.
 * All do loops are parallelized with default 'static' scheduling.
 *
 * Input :  n - grid dimension in x direction
 *          m - grid dimension in y direction
 *          alpha - Helmholtz constant (always greater than 0.0)
 *          tol   - error tolerance for iterative solver
 *          relax - Successice over relaxation parameter
 *          mits  - Maximum iterations for iterative solver
 *
 * On output
 *       : u(n,m) - Dependent variable (solutions)
 *       : f(n,m) - Right hand side function
 *************************************************************/
#define DEFAULT_DIMSIZE 256

void print_array(char *title,char *name,float *A,int n,int m)
{
  printf("%s:\n",title);
  int i;
  int j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      printf("%s[%d][%d]:%f  ",name,i,j,A[i * m + j]);
    }
    printf("\n");
  }
  printf("\n");
}
/*      subroutine initialize (n,m,alpha,dx,dy,u,f)
 ******************************************************
 * Initializes data
 * Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
 *
 ******************************************************/

void initialize(int n,int m,float alpha,float *dx,float *dy,float *u_p,float *f_p)
{
  int i;
  int j;
  int xx;
  int yy;
  float (*u)[m] = ((float (*)[m])u_p);
  float (*f)[m] = ((float (*)[m])f_p);
//double PI=3.1415926;
   *dx = (2.0 / (n - 1));
   *dy = (2.0 / (m - 1));
/* Initialize initial condition and RHS */
//#pragma omp parallel for private(xx,yy,j,i)
  for (i = 0; i < n; i++) 
    for (j = 0; j < m; j++) {
      xx = ((int )(- 1.0 + ( *dx * (i - 1))));
      yy = ((int )(- 1.0 + ( *dy * (j - 1))));
      u[i][j] = 0.0;
      f[i][j] = (- 1.0 * alpha * (1.0 - (xx * xx)) * (1.0 - (yy * yy)) - 2.0 * (1.0 - (xx * xx)) - 2.0 * (1.0 - (yy * yy)));
    }
}
/*  subroutine error_check (n,m,alpha,dx,dy,u,f)
 implicit none
 ************************************************************
 * Checks error between numerical and exact solution
 *
 ************************************************************/

void error_check(int n,int m,float alpha,float dx,float dy,float *u_p,float *f_p)
{
  int i;
  int j;
  float xx;
  float yy;
  float temp;
  float error;
  error = 0.0;
  float (*u)[m] = ((float (*)[m])u_p);
  float (*f)[m] = ((float (*)[m])f_p);
//#pragma omp parallel for private(xx,yy,temp,j,i) reduction(+:error)
  for (i = 0; i < n; i++) 
    for (j = 0; j < m; j++) {
      xx = (- 1.0 + (dx * (i - 1)));
      yy = (- 1.0 + (dy * (j - 1)));
      temp = (u[i][j] - (1.0 - (xx * xx)) * (1.0 - (yy * yy)));
      error = error + temp * temp;
    }
  error = (sqrt(error) / (n * m));
  printf("Solution Error: %2.6g\n",error);
}
void jacobi_seq(int n,int m,float dx,float dy,float alpha,float relax,float *u_p,float *f_p,float tol,int mits);
void jacobi_omp(int n,int m,float dx,float dy,float alpha,float relax,float *u_p,float *f_p,float tol,int mits);

int main(int argc,char *argv[])
{
  int status = 0;
  int n = 256;
  int m = 256;
  float alpha = 0.0543;
  float tol = 0.0000000001;
  float relax = 1.0;
  int mits = 5000;
/*fprintf(stderr, "Usage: jacobi [<n> <m> <alpha> <tol> <relax> <mits>]\n");
    fprintf(stderr, "\tn - grid dimension in x direction, default: %d\n", n);
    fprintf(stderr, "\tm - grid dimension in y direction, default: n if provided or %d\n", m);
    fprintf(stderr, "\talpha - Helmholtz constant (always greater than 0.0), default: %g\n", alpha);
    fprintf(stderr, "\ttol   - error tolerance for iterative solver, default: %g\n", tol);
    fprintf(stderr, "\trelax - Successice over relaxation parameter, default: %g\n", relax);
    fprintf(stderr, "\tmits  - Maximum iterations for iterative solver, default: %d\n", mits);*/
  if (argc == 2) {
    sscanf(argv[1],"%d",&n);
    m = n;
  }
   else if (argc == 3) {
    sscanf(argv[1],"%d",&n);
    sscanf(argv[2],"%d",&m);
  }
   else if (argc == 4) {
    sscanf(argv[1],"%d",&n);
    sscanf(argv[2],"%d",&m);
    sscanf(argv[3],"%g",&alpha);
  }
   else if (argc == 5) {
    sscanf(argv[1],"%d",&n);
    sscanf(argv[2],"%d",&m);
    sscanf(argv[3],"%g",&alpha);
    sscanf(argv[4],"%g",&tol);
  }
   else if (argc == 6) {
    sscanf(argv[1],"%d",&n);
    sscanf(argv[2],"%d",&m);
    sscanf(argv[3],"%g",&alpha);
    sscanf(argv[4],"%g",&tol);
    sscanf(argv[5],"%g",&relax);
  }
   else if (argc == 7) {
    sscanf(argv[1],"%d",&n);
    sscanf(argv[2],"%d",&m);
    sscanf(argv[3],"%g",&alpha);
    sscanf(argv[4],"%g",&tol);
    sscanf(argv[5],"%g",&relax);
    sscanf(argv[6],"%d",&mits);
  }
   else {
/* the rest of arg ignored */
  }
  printf("jacobi %d %d %g %g %g %d\n",n,m,alpha,tol,relax,mits);
  printf("------------------------------------------------------------------------------------------------------\n");
/** init the array */
  float *u = (float *)(malloc(sizeof(float ) * n * m));
  float *uomp = (float *)(malloc(sizeof(float ) * n * m));
  float *f = (float *)(malloc(sizeof(float ) * n * m));
  float dx;
/* grid spacing in x direction */
  float dy;
/* grid spacing in y direction */
  initialize(n,m,alpha,&dx,&dy,u,f);
  memcpy(uomp,u,sizeof(float ) * n * m);
  double elapsed = read_timer_ms();
  jacobi_seq(n,m,dx,dy,alpha,relax,u,f,tol,mits);
  elapsed = read_timer_ms() - elapsed;
  printf("seq elasped time(ms): %4f\n",elapsed);
  double mflops = 0.001 * mits * (n - 2) * (m - 2) * 13 / elapsed;
  printf("MFLOPS: %12.6g\n",mflops);
  puts("================");
  elapsed = read_timer_ms();
  jacobi_omp(n,m,dx,dy,alpha,relax,uomp,f,tol,mits);
  elapsed = read_timer_ms() - elapsed;
  printf("OpenMP elasped time(ms): %4f\n",elapsed);
  mflops = 0.001 * mits * (n - 2) * (m - 2) * 13 / elapsed;
  printf("MFLOPS: %12.6g\n",mflops);
//print_array("Sequential Run", "u",(REAL*)u, n, m);
  error_check(n,m,alpha,dx,dy,u,f);
  free(u);
  free(f);
  free(uomp);
  return 0;
}
/*      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,mits)
 ******************************************************************
 * Subroutine HelmholtzJ
 * Solves poisson equation on rectangular grid assuming :
 * (1) Uniform discretization in each direction, and
 * (2) Dirichlect boundary conditions
 *
 * Jacobi method is used in this routine
 *
 * Input : n,m   Number of grid points in the X/Y directions
 *         dx,dy Grid spacing in the X/Y directions
 *         alpha Helmholtz eqn. coefficient
 *         omega Relaxation factor
 *         f(n,m) Right hand side function
 *         u(n,m) Dependent variable/Solution
 *         tol    Tolerance for iterative solver
 *         mits  Maximum number of iterations
 *
 * Output : u(n,m) - Solution
 *****************************************************************/

void jacobi_seq(int n,int m,float dx,float dy,float alpha,float omega,float *u_p,float *f_p,float tol,int mits)
{
  int i;
  int j;
  int k;
  float error;
  float ax;
  float ay;
  float b;
  float resid;
  float uold[n][m];
  float (*u)[m] = ((float (*)[m])u_p);
  float (*f)[m] = ((float (*)[m])f_p);
/*
     * Initialize coefficients */
/* X-direction coef */
  ax = (1.0 / (dx * dx));
/* Y-direction coef */
  ay = (1.0 / (dy * dy));
/* Central coeff */
  b = (- 2.0 / (dx * dx) - 2.0 / (dy * dy) - alpha);
  error = (10.0 * tol);
  k = 1;
  while(k <= mits && error > tol){
    error = 0.0;
/* Copy new solution into old */
    for (i = 0; i < n; i++) 
      for (j = 0; j < m; j++) 
        uold[i][j] = u[i][j];
    for (i = 1; i < n - 1; i++) 
      for (j = 1; j < m - 1; j++) {
        resid = (ax * (uold[i - 1][j] + uold[i + 1][j]) + ay * (uold[i][j - 1] + uold[i][j + 1]) + b * uold[i][j] - f[i][j]) / b;
//printf("i: %d, j: %d, resid: %f\n", i, j, resid);
        u[i][j] = uold[i][j] - omega * resid;
        error = error + resid * resid;
      }
/* Error check */
//if (k % 500 == 0)
//    printf("Finished %d iteration with error: %g\n", k, error);
    error = (sqrt(error) / (n * m));
    k = k + 1;
  }
/*  End iteration loop */
  printf("Total Number of Iterations: %d\n",k);
  printf("Residual: %.15g\n",error);
}

void jacobi_omp(int n,int m,float dx,float dy,float alpha,float omega,float *u_p,float *f_p,float tol,int mits)
{
  int i;
  int j;
  int k;
  float error;
  float ax;
  float ay;
  float b;
  float resid;
  float *tmp = (float *)(malloc(sizeof(float ) * n * m));
  float (*uold)[m] = ((float (*)[m])tmp);
  float (*u)[m] = ((float (*)[m])u_p);
  float (*f)[m] = ((float (*)[m])f_p);
/*
     * Initialize coefficients */
/* X-direction coef */
  ax = (1.0 / (dx * dx));
/* Y-direction coef */
  ay = (1.0 / (dy * dy));
/* Central coeff */
  b = (- 2.0 / (dx * dx) - 2.0 / (dy * dy) - alpha);
  error = (10.0 * tol);
  k = 1;
  while(k <= mits && error > tol){
    error = 0.0;
//printf("===================== iteration %d ===========================\n", k);
/* Copy new solution into old */
    for (i = 0; i < n; i++) {
      for (j = 0; j <= m - 1; j += 16) {
        float *__ptr39 = uold[i];
        float *__ptr40 = u[i];
        __m512 __vec41 = _mm512_loadu_ps(&__ptr40[j]);
        _mm512_storeu_ps(&__ptr39[j],__vec41);
      }
    }
    for (i = 1; i < n - 1; i++) {
      __m512 __vec0 = _mm512_set1_ps(ax);
      __m512 __vec7 = _mm512_set1_ps(ay);
      __m512 __vec15 = _mm512_set1_ps(b);
      __m512 __vec23 = _mm512_set1_ps(b);
      __m512 __part25 = _mm512_setzero_ps();
      __m512 __vec29 = _mm512_set1_ps(omega);
      __m512 __vec30 = _mm512_set1_ps(resid);
      __m512 __vec33 = _mm512_set1_ps(error);
      __m512 __vec34 = _mm512_set1_ps(resid);
      __m512 __vec35 = _mm512_set1_ps(resid);
      __m512 __part38 = _mm512_setzero_ps();
      for (j = 1; j <= m - 1 - 1; j += 16) {
        float *__ptr1 = uold[i - 1];
        __m512 __vec2 = _mm512_loadu_ps(&__ptr1[j]);
        float *__ptr3 = uold[i + 1];
        __m512 __vec4 = _mm512_loadu_ps(&__ptr3[j]);
        __m512 __vec5 = _mm512_add_ps(__vec4,__vec2);
        __m512 __vec6 = _mm512_mul_ps(__vec5,__vec0);
        float *__ptr8 = uold[i];
        __m512 __vec9 = _mm512_loadu_ps(&__ptr8[j - 1]);
        float *__ptr10 = uold[i];
        __m512 __vec11 = _mm512_loadu_ps(&__ptr10[j + 1]);
        __m512 __vec12 = _mm512_add_ps(__vec11,__vec9);
        __m512 __vec13 = _mm512_mul_ps(__vec12,__vec7);
        __m512 __vec14 = _mm512_add_ps(__vec13,__vec6);
        float *__ptr16 = uold[i];
        __m512 __vec17 = _mm512_loadu_ps(&__ptr16[j]);
        __m512 __vec18 = _mm512_mul_ps(__vec17,__vec15);
        __m512 __vec19 = _mm512_add_ps(__vec18,__vec14);
        float *__ptr20 = f[i];
        __m512 __vec21 = _mm512_loadu_ps(&__ptr20[j]);
        __m512 __vec22 = _mm512_sub_ps(__vec21,__vec19);
        __m512 __vec24 = _mm512_div_ps(__vec23,__vec22);
        __part25 = _mm512_add_ps(__part25,__vec24);
        float *__ptr26 = u[i];
        float *__ptr27 = uold[i];
        __m512 __vec28 = _mm512_loadu_ps(&__ptr27[j]);
        __m512 __vec31 = _mm512_mul_ps(__vec30,__vec29);
        __m512 __vec32 = _mm512_sub_ps(__vec31,__vec28);
        _mm512_storeu_ps(&__ptr26[j],__vec32);
        __m512 __vec36 = _mm512_mul_ps(__vec35,__vec34);
        __m512 __vec37 = _mm512_add_ps(__vec36,__vec33);
        __part38 = _mm512_add_ps(__part38,__vec37);
      }
      __m256 __buf3 = _mm512_extractf32x8_ps(__part38,0);
      __m256 __buf4 = _mm512_extractf32x8_ps(__part38,1);
      __buf4 = _mm256_add_ps(__buf3,__buf4);
      __buf4 = _mm256_hadd_ps(__buf4,__buf4);
      __buf4 = _mm256_hadd_ps(__buf4,__buf4);
      float __buf5[8];
      _mm256_storeu_ps(&__buf5,__buf4);
      error = __buf5[0] + __buf5[6];
      __m256 __buf0 = _mm512_extractf32x8_ps(__part25,0);
      __m256 __buf1 = _mm512_extractf32x8_ps(__part25,1);
      __buf1 = _mm256_add_ps(__buf0,__buf1);
      __buf1 = _mm256_hadd_ps(__buf1,__buf1);
      __buf1 = _mm256_hadd_ps(__buf1,__buf1);
      float __buf2[8];
      _mm256_storeu_ps(&__buf2,__buf1);
      resid = __buf2[0] + __buf2[6];
    }
/* Error check */
//if (k % 500 == 0)
//    printf("Finished %d iteration with error: %g\n", k, error);
    error = (sqrt(error) / (n * m));
    k = k + 1;
  }
/*  End iteration loop */
  printf("Total Number of Iterations: %d\n",k);
  printf("Residual: %.15g\n",error);
  free(tmp);
}
