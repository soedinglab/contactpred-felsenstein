#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "felsenstein_simd.h"

int main() {
  int i = 0;
  int j = 1;

  int N = 64;
  int L = 5;
  uint8_t* msa = (uint8_t*) malloc(sizeof(u_int8_t)*N*L);
  for(int i = 0; i < L; i++) {
    for(int j = 0; j < N; j++) {
      if (j < N / 2) {
        msa[j*L + i] = 1;
      } else {
        msa[j*L + i] = 2;
      }
    }
  }

  c_float_t t1 = 0.2;
  c_float_t phi1 = exp(-t1);
  c_float_t t2 = 0.2;
  c_float_t phi2 = exp(-t2);

  Node* nodes = (Node*) malloc(sizeof(Node)*(2*N - 1));
  for(int n = 0; n < N - 1; n++) {
    nodes[n].seq_id = -n - 1;
    nodes[n].left = nodes + 2*n + 1;
    nodes[n].right = nodes + 2*n + 2;
    nodes[n].phi_left = phi1;
    nodes[n].phi_right = phi2;
  }
  for(int n = N - 1; n < 2*N - 1; n++) {
    nodes[n].left = NULL;
    nodes[n].right = NULL;
    nodes[n].seq_id = n - N + 1;
  }

  int A_i = 5;
  int A_j = 4;
  int A_i_p_A_j = A_i + A_j;
  int AA_ij = A_i * A_j;

  c_float_t* x = (c_float_t*) calloc(A_i_p_A_j + AA_ij, sizeof(c_float_t));

  for(int idx = 0; idx < A_i_p_A_j; idx++) {
    x[idx] = log0;
  }
  x[0] = 1;
  x[1] = 0;
  x[2] = 0;
  x[A_i + 0] = 0;
  x[A_i + 1] = 0;
  x[A_i + 2] = 0;

  for(int a = 0; a < 3; a++) {
    for(int b = 0; b < 3; b++) {
      x[A_i_p_A_j + a * A_j + b] = a < b ? 1 : 2;
    }
  }

  c_float_t* grad = (c_float_t*) malloc(sizeof(c_float_t)*(A_i_p_A_j + AA_ij));

  Constants* consts = malloc(sizeof(Constants));
  consts->phylo_tree = nodes;
  consts->msa = msa;
  consts->L = L;
  consts->i = i;
  consts->j = j;
  consts->A_i = A_i;
  consts->A_j = A_j;
  consts->A_i_p_A_j = A_i_p_A_j;
  consts->AA_ij = AA_ij;
  initialize_constants(consts);

  Buffer* buffer = malloc(sizeof(Buffer));
  initialize_buffer(buffer, consts);

  c_float_t fx = calculate_fx_grad(x, grad, consts, buffer);
  printf("fx= %e\n", fx);

  c_float_t epsilon = 1e-4;
  int pos = 0;
  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    calculate_fx_grad(x, grad, consts, buffer);
    c_float_t target_grad = grad[pos];

    x[pos] += epsilon;
    c_float_t fx_fwd = calculate_fx_grad(x, grad, consts, buffer);
    x[pos] -= 2*epsilon;
    c_float_t fx_rev = calculate_fx_grad(x, grad, consts, buffer);
    x[pos] += epsilon;
    printf("v|ic=%d %e / %e\n", lc, target_grad, (fx_fwd - fx_rev) / (2*epsilon));
    pos++;
  }

  for(int c = 0; c < A_i; c++) {
    for(int d = 0; d < A_j; d++) {
      calculate_fx_grad(x, grad, consts, buffer);
      c_float_t target_grad = grad[pos];

      x[pos] += epsilon;
      c_float_t fx_fwd = calculate_fx_grad(x, grad, consts, buffer);
      x[pos] -= 2*epsilon;
      c_float_t fx_rev = calculate_fx_grad(x, grad, consts, buffer);
      x[pos] += epsilon;
      printf("w|c=%d|d=%d %e / %e\n", c, d, target_grad, (fx_fwd - fx_rev) / (2*epsilon));
      pos++;
    }
  }

  // cleanup
  free(x);
  free(grad);

  deinitialize_buffer(buffer);
  free(buffer);

  deinitialize_constants(consts);
  free(consts->msa);
  free(consts->phylo_tree);
  free(consts);
}
