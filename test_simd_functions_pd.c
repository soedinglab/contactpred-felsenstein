#include "simd_functions_pd.h"
#include "simd.h"


void printf64(simdf64 x) {
  double y_arr[4];
  simdf64_store(y_arr, x);
  printf("%.17g\n", y_arr[0]);
  printf("%.17g\n", y_arr[1]);
  printf("%.17g\n", y_arr[2]);
  printf("%.17g\n", y_arr[3]);
}

int main() {

  simdf64 x  = simdf64_set(-3);
  simdf64 pow_x = simdf64_pow2(x);
  printf64(pow_x);

  printf("\n");

  simdf64 y = simdf64_set(32);
  simdf64 log2_x = simdf64_log2(y);
  printf64(log2_x);
}

