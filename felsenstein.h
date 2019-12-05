#ifndef FELSENSTEIN_FELSENSTEIN_H
#define FELSENSTEIN_FELSENSTEIN_H

#include <stdint.h>
#include <math.h>

#define c_f0 0.0
#define N_COL 2
#define A 20
#define AA 400
#define AAA 8000
#define AAAA 160000



typedef double c_float_t;

typedef struct SignedLogExp {
  int8_t sign;
  c_float_t result;
} SignedLogExp;


static inline c_float_t logsumexp2(c_float_t log_x, c_float_t log_y) {
  c_float_t logsumexp;
  if(log_x > log_y) {
    logsumexp = log_x + log(1 + exp(log_y - log_x));
  } else {
    logsumexp = log_y + log(exp(log_x - log_y) + 1);
  }
  return logsumexp;
}

static inline SignedLogExp signed_logsumexp2(c_float_t log_x, int8_t sign_x, c_float_t log_y, int8_t sign_y) {
  c_float_t logsumexp;
  int8_t sign;
  if(log_x > log_y) {
    c_float_t exp_part = sign_x + sign_y * exp(log_y - log_x);
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_x + log(sign*exp_part);
  } else {
    c_float_t exp_part = sign_x * exp(log_x - log_y) + sign_y;
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_y + log(sign*exp_part);
  }
  SignedLogExp result = {sign, logsumexp};
  return result;
}

static inline c_float_t logsumexp3(c_float_t log_x, c_float_t log_y, c_float_t log_z) {
  c_float_t logsumexp;
  if(log_x > log_y && log_x > log_z) {
    logsumexp = log_x + log(1 + exp(log_y - log_x) + exp(log_z - log_x));
  } else if(log_y > log_x && log_y > log_z) {
    logsumexp = log_y + log(exp(log_x - log_y) + 1 + exp(log_z - log_y));
  } else {
    logsumexp = log_z + log(exp(log_x - log_z) + exp(log_y - log_z) + 1);
  }
  return logsumexp;
}

static inline SignedLogExp signed_logsumexp3(c_float_t log_x, int8_t sign_x, c_float_t log_y, int8_t sign_y, c_float_t log_z, int8_t sign_z) {
  c_float_t logsumexp;
  int8_t sign;
  if(log_x > log_y && log_x > log_z) {
    c_float_t exp_part = sign_x + sign_y*exp(log_y - log_x) + sign_z*exp(log_z - log_x);
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_x + log(sign*exp_part);
  } else if(log_y > log_x && log_y > log_z) {
    c_float_t exp_part = sign_x*exp(log_x - log_y) + sign_y + sign_z*exp(log_z - log_y);
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_y + log(sign*exp_part);
  } else {
    c_float_t exp_part = sign_x*exp(log_x - log_z) + sign_y*exp(log_y - log_z) + sign_z;
    sign = exp_part >= 0 ? 1 : -1;
    logsumexp = log_z + log(sign*exp_part);
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
    exp_sum += exp(log_vals[i] - max);
  }
  return max + log(exp_sum);
}

static inline SignedLogExp signed_logsumexp_n(c_float_t* log_vals, int8_t* signs, int n) {
  c_float_t max = log_vals[0];
  for(int i = 1; i < n; i++) {
    max = (log_vals[i] > max) ? log_vals[i] : max;
  }
  c_float_t exp_sum = 0;
  for(int i = 0; i < n; i++) {
    exp_sum += signs[i] * exp(log_vals[i] - max);
  }
  int8_t sign = exp_sum >= 0 ? 1 : -1;
  SignedLogExp result = {sign, max + log(exp_sum * sign)};
  return result;
}

static inline c_float_t logsumexp_quot3(c_float_t a, c_float_t b, c_float_t c, c_float_t d, c_float_t e, c_float_t f) {
  // computes log[ (e^a + e^b + e^c) / (e^d + e^e + e^f) ]

  c_float_t num_max;
  if(a > b && a > c)
    num_max = a;
  else if(b > a && b > c)
    num_max = b;
  else
    num_max = c;

  c_float_t denom_max;
  if(d > e && d > f)
    denom_max = d;
  else if(e > d && e > f)
    denom_max = e;
  else
    denom_max = f;

  c_float_t num_res = log(exp(a - num_max) + exp(b - num_max) + exp(c - num_max));
  c_float_t denom_res = log(exp(d - denom_max) + exp(e - denom_max) + exp(f - denom_max));

  return num_max - denom_max + num_res - denom_res;
}

static inline c_float_t logsumexp_quotn(c_float_t* num, int num_len, c_float_t* denom, int denom_len) {
  c_float_t num_max = num[0];
  for(int i = 1; i < num_len; i++) {
    num_max = (num[i] > num_max) ? num[i] : num_max;
  }
  c_float_t denom_max = num[0];
  for(int i = 1; i < denom_len; i++) {
    denom_max = (denom[i] > denom_max) ? denom[i] : denom_max;
  }
  c_float_t exp_num = 0;
  for(int i = 0; i < num_len; i++) {
    exp_num += exp(num[i] - num_max);
  }
  c_float_t exp_denom = 0;
  for(int i = 0; i < denom_len; i++) {
    exp_denom += exp(denom[i] - denom_max);
  }
  return num_max - denom_max + log(exp_num) - log(exp_denom);
}

static inline c_float_t logsumexp_quot_2num1denom(c_float_t num1, c_float_t num2, c_float_t denom) {
  if(num1 > num2) {
    return num1 - denom + log(1 + exp(num2 - num1));
  } else {
    return num2 - denom + log(exp(num1 - num2) + 1);
  }
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
  c_float_t* single_aa_frequencies;

  Node* phylo_tree;
  uint8_t* msa;
  int i;
  int j;

} Constants;

void initialize_node(Node* node);
void deinitialize_node(Node* node);
void initialize_leaf(Node* leaf, Constants* consts);

void initialize_constants(Constants* consts);
void precalculate_constants(Constants* consts, c_float_t* v, c_float_t* w);
void deinitialize_constants(Constants* consts);

void initialize_buffer(NodeBuffer*);
void precompute_buffer(NodeBuffer* buffer, NodePrecomputation* data, Constants* consts);
void deinitialize_buffer(NodeBuffer*);

c_float_t calculate_fx_grad(c_float_t* x, c_float_t* grad, Constants* consts, Buffer* buf);

#endif //FELSENSTEIN_FELSENSTEIN_H