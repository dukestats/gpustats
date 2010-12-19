#include <stdio.h>
#include <stdlib.h>

#include "gpustats_common.h"
#include "kernels.h"

REAL* make_random_matrix(int n, int k);
REAL* make_random_cov(int k);

REAL* make_random_matrix(int n, int k) {
  REAL* result = (REAL*) malloc(n * k);

  return result;
}

REAL* make_random_cov(int k) {
  REAL* result = (REAL*) malloc(k * k);

}

int main(int argc, char *argv[])
{
	printf("it's alive!\n");
	return 0;
}
