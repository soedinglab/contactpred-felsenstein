#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "felsenstein_lin.h"

int main() {
  int i = 0;
  int j = 1;

  int N = 4;
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

  c_float_t* x = (c_float_t*) calloc(N_COL*A + A*A, sizeof(c_float_t));
  for(int idx = 0; idx < N_COL*A; idx++) {
    x[idx] = log0;
  }
  x[0] = 1;
  x[1] = 0;
  x[2] = 0;
  x[A + 0] = 0;
  x[A + 1] = 0;
  x[A + 2] = 0;

  for(int a = 0; a < 3; a++) {
    for(int b = 0; b < 3; b++) {
      x[N_COL*A + a * A + b] = a < b ? 1 : 2;
    }
  }

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

  c_float_t* grad = (c_float_t*) malloc(sizeof(c_float_t)*(N_COL*A + A*A));

  Constants* consts = malloc(sizeof(Constants));
  consts->phylo_tree = nodes;
  consts->msa = msa;
  consts->L = L;
  consts->i = i;
  consts->j = j;
  consts->A_i = A;
  consts->A_j = A;
  consts->A_i_p_A_j = A + A;
  consts->AA_ij = A*A;

  initialize_constants(consts);

  Buffer* buffer = malloc(sizeof(Buffer));
  buffer->left = malloc(sizeof(NodeBuffer));
  initialize_buffer(buffer->left, consts);
  buffer->right = malloc(sizeof(NodeBuffer));
  initialize_buffer(buffer->right, consts);

  c_float_t fx = calculate_fx_grad(x, grad, consts, buffer);
  printf("fx= %e\n", fx);

  //exit(-2);
  c_float_t epsilon = 1e-9;
  int pos = 0;


  for(int lc = 0; lc < N_COL*A; lc++) {
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


  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
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

  deinitialize_buffer(buffer->left);
  deinitialize_buffer(buffer->right);
  free(buffer->left);
  free(buffer->right);
  free(buffer);

  deinitialize_constants(consts);
  free(consts->msa);
  free(consts->phylo_tree);
  free(consts);
}
