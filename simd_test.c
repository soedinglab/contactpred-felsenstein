
//
// Created by Christian Roth on 21/01/2020.
//
#include <stdio.h>
#include "simd.h"
#include "simd_functions.h"
#include <string.h>

#define pad_float(N)  (N + (VECSIZE_FLOAT-1))/VECSIZE_FLOAT*VECSIZE_FLOAT


int main() {
    size_t N = 8;
    size_t N_pad = pad_float(N);
    float* x = (float*) malloc_simd_float(N_pad*sizeof(float));
    float* y = (float*) malloc_simd_float(N_pad*sizeof(float));
    float* x_signs = (float*) malloc_simd_float(N_pad*sizeof(float));
    float* y_signs = (float*) malloc_simd_float(N_pad*sizeof(float));

    for(int i = 0; i < N; i++) {
      x[i] = i;
      y[i] = N - 1 - i;
      x_signs[i] = 1;
      y_signs[i] = -1;
    }

    float* out = (float*) malloc_simd_float(N_pad*sizeof(float));
    float* out_signs = (float*) malloc_simd_float(N_pad*sizeof(float));
    signedlogsumexp2_array(out, out_signs, x, x_signs, y, y_signs, N);
    for(int i = 0; i < N_pad; i++) {
      //printf("%f ", out[i]);
    }
    for(int i = 0; i < N_pad; i++) {
      //printf("%f ", out_signs[i]);
    }

    free(x);
    free(y);

    int A_i = 2;
    int A_j = 2;
    int padded_AA_ij = pad_float(A_i*A_j);

    float* data_block = malloc_simd_float(A_i*A_j*padded_AA_ij*sizeof(float));
    float* sign_block = malloc_simd_float(A_i*A_j*padded_AA_ij*sizeof(float));
    float (*data_3d)[A_j][padded_AA_ij] = (float (*)[A_j][padded_AA_ij]) data_block;
    float (*sign_3d)[A_j][padded_AA_ij] = (float (*)[A_j][padded_AA_ij]) sign_block;

    int i = 0;
    for(int c_p = 0; c_p < A_i; c_p++) {
      for(int d_p = 0; d_p < A_j; d_p++) {
        for(int cd = 0; cd < A_i*A_j; cd++) {
          data_3d[c_p][d_p][cd] = i--;
          sign_3d[c_p][d_p][cd] = -1;
        }
      }
    }

    float* result =  malloc_simd_float(padded_AA_ij*sizeof(float));
    float* result_sign = malloc_simd_float(padded_AA_ij*sizeof(float));
    logsumexp_matrix_ax01(result, result_sign, A_i, A_j, padded_AA_ij, data_3d, sign_3d);

    for(int i = 0; i < A_i*A_j; i++) {
      printf("%f ", result[i]);
    }
    printf("\n");
  for(int i = 0; i < A_i*A_j; i++) {
    printf("%f ", result_sign[i]);
  }


}

