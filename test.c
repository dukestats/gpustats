#include <stdio.h>
#include <stdlib.h>

#include "mvnpdf.h"

float* make_random_matrix(int n, int k);
float* make_random_cov(int k);

void fill_random_array(float* arr, int size){
  for (int i = 0; i < size; ++i){
	arr[i] = rand();
  }
}

float* make_random_matrix(int n, int k) {
  int size = n * k * sizeof(float);
  float* result = (float*) malloc(size);
  fill_random_array(result, size);
  return result;
}

float* make_random_cov(int k) {
  int size = k * (k + 1) / 2;
  float* lower = (float*) malloc(size * sizeof(float));
  fill_random_array(lower, size);
}

int main(int argc, char *argv[])
{
  int k = 4;
  int n = 10;

  float* cov = make_random_matrix(k, k);

  printf("it's alive!\n");
  return 0;
}
