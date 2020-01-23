#ifndef FELSENSTEIN_FELSENSTEIN_H
#define FELSENSTEIN_FELSENSTEIN_H

#include <stdint.h>
#include <math.h>

#include "simd.h"

#define c_f0 0.0
#define N_COL 2
#define A 20
#define AA 400
#define AAA 8000
#define AAAA 160000
#define log0 -1000

typedef float c_float_t;

typedef struct SignedLogExp {
  int8_t sign;
  c_float_t result;
} SignedLogExp;


static inline void initialize_array(c_float_t* arr, c_float_t value, int length) {
  for(int i = 0; i < length; i++) {
    arr[i] = value;
  }
}

static inline c_float_t logsumexp2(c_float_t log_x, c_float_t log_y) {
  c_float_t logsumexp;
  if(log_x > log_y) {
    c_float_t exp_part = 1 + powf(2, log_y - log_x);
    logsumexp = log_x + ((exp_part != 0) ? log2f(exp_part) : log0);
  } else {
    c_float_t exp_part = powf(2, log_x - log_y) + 1;
    logsumexp = log_y + ((exp_part != 0) ? log2(exp_part) : log0);
  }
  return logsumexp;
}

static inline SignedLogExp signed_logsumexp2(c_float_t log_x, int8_t sign_x, c_float_t log_y, int8_t sign_y) {
  c_float_t logsumexp;
  int8_t sign;
  if(log_x > log_y) {
    c_float_t exp_part = sign_x + sign_y * powf(2, log_y - log_x);
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_x + ((exp_part != 0) ? log2(sign*exp_part) : log0);
  } else {
    c_float_t exp_part = sign_x * powf(2, log_x - log_y) + sign_y;
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_y + ((exp_part != 0) ? log2(sign*exp_part) : log0);
  }
  SignedLogExp result = {sign, logsumexp};
  return result;
}

static inline c_float_t logsumexp3(c_float_t log_x, c_float_t log_y, c_float_t log_z) {
  c_float_t logsumexp;
  if(log_x > log_y && log_x > log_z) {
    c_float_t exp_part = 1 + powf(2, log_y - log_x) + powf(2, log_z - log_x);
    logsumexp = log_x + ((exp_part != 0) ? log2(exp_part) : log0);
  } else if(log_y > log_x && log_y > log_z) {
    c_float_t exp_part = powf(2, log_x - log_y) + 1 + powf(2, log_z - log_y);
    logsumexp = log_y + ((exp_part != 0) ? log2(exp_part) : log0);
  } else {
    c_float_t exp_part = powf(2, log_x - log_z) + powf(2, log_y - log_z) + 1;
    logsumexp = log_z + ((exp_part != 0) ? log2(exp_part) : log0);
  }
  return logsumexp;
}

static inline SignedLogExp signed_logsumexp3(c_float_t log_x, int8_t sign_x, c_float_t log_y, int8_t sign_y, c_float_t log_z, int8_t sign_z) {
  c_float_t logsumexp;
  int8_t sign;
  if(log_x > log_y && log_x > log_z) {
    c_float_t exp_part = sign_x + sign_y*powf(2, log_y - log_x) + sign_z*powf(2, log_z - log_x);
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_x + ((exp_part != 0) ? log2(sign*exp_part) : log0);
  } else if(log_y > log_x && log_y > log_z) {
    c_float_t exp_part = sign_x*powf(2, log_x - log_y) + sign_y + sign_z*powf(2, log_z - log_y);
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_y + ((exp_part != 0) ? log2(sign*exp_part) : log0);
  } else {
    c_float_t exp_part = sign_x*powf(2, log_x - log_z) + sign_y*powf(2, log_y - log_z) + sign_z;
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_z + ((exp_part != 0) ? log2(sign*exp_part) : log0);
  }
  SignedLogExp result = {sign, logsumexp};
  return result;
}

static inline c_float_t logsumexpn(c_float_t* log_vals, int n) {
  if( n < 2) {
    return log_vals[0];
  }
  c_float_t max = log_vals[0];
  for(int i = 1; i < n; i++) {
    max = (log_vals[i] > max) ? log_vals[i] : max;
  }
  c_float_t exp_sum = 0;
  for(int i = 0; i < n; i++) {
    exp_sum += powf(2, log_vals[i] - max);
  }
  return max + ((exp_sum != 0) ? log2(exp_sum) : log0);
}

static inline SignedLogExp signed_logsumexp_n(c_float_t* log_vals, int8_t* signs, int n) {
  c_float_t max = log_vals[0];
  for(int i = 1; i < n; i++) {
    max = (log_vals[i] > max) ? log_vals[i] : max;
  }
  c_float_t exp_sum = 0;
  for(int i = 0; i < n; i++) {
    exp_sum += signs[i] * powf(2, log_vals[i] - max);
  }
  int8_t sign = exp_sum >= 0 ? 1 : -1;
  SignedLogExp result = {sign, max + ((exp_sum != 0) ? log2(exp_sum * sign) : log0)};
  return result;
}


typedef struct NodePrecomputation {
  // dim: A*A [a, b]
  c_float_t* Ln_ab;
  // dim: 2*A*A*A [i, c, a, b]
  c_float_t* dv_Ln_ab;
  int8_t* dv_Ln_ab_signs;
  // dim: A*A*A*A [c, d, a, b]
  c_float_t* dw_Ln_ab;
  int8_t* dw_Ln_ab_signs;

} NodePrecomputation;

typedef struct Node {
  struct Node* left;
  struct Node* right;

  int seq_id;
  c_float_t phi_left;
  c_float_t phi_right;

  NodePrecomputation* data;

} Node;

typedef struct NodeBuffer {
  // dim: scalar
  c_float_t Ln;
  // dim: A [a]
  c_float_t* Ln_ia;
  // dim: A [b]
  c_float_t* Ln_jb;

  // dim: 2*A [i, c]
  c_float_t* dv_Ln;
  int8_t* dv_Ln_signs;

  // dim: 2*A*A [i, c, a]
  c_float_t* dv_Ln_ia;
  int8_t* dv_Ln_ia_signs;

  // dim: 2*A*A [i, c, b]
  c_float_t* dv_Ln_jb;
  int8_t* dv_Ln_jb_signs;

  // dim: A*A [c, d]
  c_float_t* dw_Ln;
  int8_t* dw_Ln_signs;

  // dim: A*A*A [c, d, a]
  c_float_t* dw_Ln_ia;
  int8_t* dw_Ln_ia_signs;

  // dim: A*A*A [c, d, b]
  c_float_t* dw_Ln_jb;
  int8_t* dw_Ln_jb_signs;

} NodeBuffer;

typedef struct Buffer {
  NodeBuffer* left;
  NodeBuffer* right;
} Buffer;

typedef struct Constants {
  int L;

  c_float_t* p_ab;
  c_float_t* dw_p_ab;
  int8_t* dw_p_ab_signs;
  c_float_t* dv_p_ab;
  int8_t* dv_p_ab_signs;

  c_float_t* p_ij_cond;
  c_float_t* dw_p_ij_cond;
  int8_t* dw_p_ij_cond_signs;
  c_float_t* dv_p_ij_cond;
  int8_t* dv_p_ij_cond_signs;

  c_float_t* p_ji_cond;
  c_float_t* dw_p_ji_cond;
  int8_t* dw_p_ji_cond_signs;
  c_float_t* dv_p_ji_cond;
  int8_t* dv_p_ji_cond_signs;

  Node* phylo_tree;
  uint8_t* msa;

  int A_a;
  int A_b;
  int A_a_p_A_b;
  int AA_ab;

  int i;
  int j;

} Constants;

void initialize_node(Node* node, Constants* consts);
void deinitialize_node(Node* node);
void initialize_leaf(Node* leaf, Constants* consts);

void initialize_constants(Constants* consts);
void precalculate_constants(Constants* consts, c_float_t* v, c_float_t* w);
void deinitialize_constants(Constants* consts);

void initialize_buffer(NodeBuffer*, Constants* consts);
void precompute_buffer(NodeBuffer* buffer, NodePrecomputation* data, Constants* consts);
void deinitialize_buffer(NodeBuffer*);

c_float_t calculate_fx_grad(c_float_t* x, c_float_t* grad, Constants* consts, Buffer* buf);

#endif //FELSENSTEIN_FELSENSTEIN_H