#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "felsenstein.h"

int main() {
  int i = 0;
  int j = 1;

  int N = 4;
  int L = 5;
  uint8_t* msa = (uint8_t*) malloc(sizeof(u_int8_t)*N*L);
  for(int i = 0; i < L; i++) {
    msa[0*L + i] = 0;
    msa[1*L + i] = 2;
    msa[2*L + i] = 1;
    msa[3*L + i] = 1;
  }

  int A_a = 3;
  int A_b = 3;
  int A_max = A_a > A_b ? A_a : A_b;
  int A_a_p_A_b = A_a + A_b;
  int AA_ab = A_a * A_b;

  c_float_t t1 = 0.2;
  c_float_t phi1 = exp(-t1);
  c_float_t t2 = 0.2;
  c_float_t phi2 = exp(-t2);

  c_float_t* aa_freqs = (c_float_t*) calloc(A_max, sizeof(c_float_t));

  c_float_t aa_counts[A_max];
  for(int a = 0; a < A_max; a++) {
    aa_counts[a] = 1e-9;
  }

  for(int a = 0; a < N*L; a++) {
    aa_counts[msa[a]] += 1;
  }

  c_float_t norm = 0;
  for(int a = 0; a < A_max; a++) {
    norm += aa_counts[a];
  }

  for(int a = 0; a < A_max; a++) {
    aa_freqs[a] = log(aa_counts[a] / norm);
  }


  c_float_t* x = (c_float_t*) calloc(A_a_p_A_b + AA_ab, sizeof(c_float_t));
  for(int idx = 0; idx < A_a_p_A_b; idx++) {
    x[idx] = log0;
  }
  x[0] = 0;
  x[1] = 0;
  x[2] = 0;
  x[A_a + 0] = 0;
  x[A_a + 1] = 0;
  x[A_a + 2] = 0;


  c_float_t* grad = (c_float_t*) malloc(sizeof(c_float_t)*(A_a_p_A_b + AA_ab));

  Node ll_node;
  ll_node.seq_id = 0;
  ll_node.left = NULL;
  ll_node.right = NULL;

  Node lr_node;
  lr_node.seq_id = 1;
  lr_node.left = NULL;
  lr_node.right = NULL;

  Node rl_node;
  rl_node.seq_id = 2;
  rl_node.left = NULL;
  rl_node.right = NULL;

  Node rr_node;
  rr_node.seq_id = 3;
  rr_node.left = NULL;
  rr_node.right = NULL;

  Node left_node;
  left_node.left = &ll_node;
  left_node.right = &lr_node;
  left_node.phi_left = phi1;
  left_node.phi_right = phi2;
  left_node.seq_id = -2;

  Node right_node;
  right_node.left = &rl_node;
  right_node.right = &rr_node;
  right_node.phi_left = phi1;
  right_node.phi_right = phi2;
  right_node.seq_id = -3;

  Node* root = malloc(sizeof(Node));
  root->phi_left = phi1;
  root->phi_right = phi2;
  root->left = &left_node;
  root->right = &right_node;
  root->seq_id = -1;

  Constants* consts = malloc(sizeof(Constants));
  consts->single_aa_frequencies = aa_freqs;
  consts->phylo_tree = root;
  consts->msa = msa;
  consts->L = L;
  consts->i = i;
  consts->j = j;
  consts->A_a = A_a;
  consts->A_b = A_b;
  consts->A_a_p_A_b = A_a_p_A_b;
  consts->AA_ab = AA_ab;
  initialize_constants(consts);

  Buffer* buffer = malloc(sizeof(Buffer));
  buffer->left = malloc(sizeof(NodeBuffer));
  initialize_buffer(buffer->left, consts);
  buffer->right = malloc(sizeof(NodeBuffer));
  initialize_buffer(buffer->right, consts);

  c_float_t fx = calculate_fx_grad(x, grad, consts, buffer);
  printf("fx= %e\n", fx);

  c_float_t epsilon = 1e-9;
  int pos = 0;
  for(int lc = 0; lc < A_a_p_A_b; lc++) {
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

  for(int c = 0; c < A_a; c++) {
    for(int d = 0; d < A_b; d++) {
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
}
