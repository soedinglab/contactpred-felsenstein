#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "felsenstein.h"

int main() {
  int i = 0;
  int j = 1;

  int N = 2;
  int L = 5;
  uint8_t* msa = (uint8_t*) malloc(sizeof(u_int8_t)*N*L);
  for(int i = 0; i < L; i++) {
    msa[0*L + i] = 1;
    msa[1*L + i] = 2;
  }

  c_float_t t1 = 10;
  c_float_t phi1 = exp(-t1);
  c_float_t t2 = 15;
  c_float_t phi2 = exp(-t2);

  c_float_t* aa_freqs = (c_float_t*) calloc(A, sizeof(c_float_t));

  for(int a = 0; a < A; a++) {
    aa_freqs[a] = 1e-5;
  }
  aa_freqs[1] += 0.5;
  aa_freqs[2] += 0.5;

  c_float_t norm = 0;
  for(int a = 0; a < A; a++) {
    norm += aa_freqs[a];
  }
  for(int a = 0; a < A; a++) {
    aa_freqs[a] = aa_freqs[a] / norm;
  }

  c_float_t* x = (c_float_t*) malloc(sizeof(c_float_t)*(N_COL*A + A*A));
  for(int idx = 0; idx < N_COL*A + A*A; idx++) {
    x[idx] = (c_float_t)rand() / (c_float_t)RAND_MAX;
  }
  c_float_t* grad = (c_float_t*) malloc(sizeof(c_float_t)*(N_COL*A + A*A));

  Node* left_node = malloc(sizeof(Node));
  left_node->seq_id = 0;
  left_node->left = NULL;
  left_node->right = NULL;
  Node* right_node = malloc(sizeof(Node));
  right_node->left = NULL;
  right_node->right = NULL;
  right_node->seq_id = 1;

  Node* root = malloc(sizeof(Node));
  root->phi_left = phi1;
  root->phi_right = phi2;
  root->left = left_node;
  root->right = right_node;

  Constants* consts = malloc(sizeof(Constants));
  consts->single_aa_frequencies = aa_freqs;
  consts->phylo_tree = root;
  consts->msa = msa;
  consts->L = L;
  consts->i = i;
  consts->j = j;
  initialize_constants(consts);

  Buffer* buffer = malloc(sizeof(Buffer));
  buffer->left = malloc(sizeof(NodeBuffer));
  initialize_buffer(buffer->left);
  buffer->right = malloc(sizeof(NodeBuffer));
  initialize_buffer(buffer->right);


  c_float_t epsilon = 1e-9;
  int pos = 0;
  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      calculate_fx_grad(x, grad, consts, buffer);
      c_float_t target_grad = grad[pos];

      x[pos] += epsilon;
      c_float_t fx_fwd = calculate_fx_grad(x, grad, consts, buffer);
      x[pos] -= 2*epsilon;
      c_float_t fx_rev = calculate_fx_grad(x, grad, consts, buffer);
      x[pos] += epsilon;
      printf("v|i=%d|c=%d %e / %e\n", l, c, target_grad, (fx_fwd - fx_rev) / (2*epsilon));
      pos++;
    }
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
  free(consts->single_aa_frequencies);
  free(consts->msa);
  free(consts->phylo_tree);
  free(consts);

  free(left_node);
  free(right_node);

}
