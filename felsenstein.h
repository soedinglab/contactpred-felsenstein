#ifndef FELSENSTEIN_FELSENSTEIN_H
#define FELSENSTEIN_FELSENSTEIN_H

#include <stdint.h>

#define c_f0 0.0
#define N_COL 2
#define A 20
#define AA 400
#define AAA 8000
#define AAAA 160000

typedef double c_float_t;

typedef struct NodePrecomputation {
  // dim: A*A [a, b]
  c_float_t* Ln_ab;
  // dim: 2*A*A*A [i, c, a, b]
  c_float_t* dv_Ln_ab;
  // dim: A*A*A*A [c, d, a, b]
  c_float_t* dw_Ln_ab;

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
  // dim: 2*A*A [i, c, a]
  c_float_t* dv_Ln_ia;
  // dim: 2*A*A [i, c, b]
  c_float_t* dv_Ln_jb;

  // dim: A*A [c, d]
  c_float_t* dw_Ln;
  // dim: A*A*A [c, d, a]
  c_float_t* dw_Ln_ia;
  // dim: A*A*A [c, d, b]
  c_float_t* dw_Ln_jb;

} NodeBuffer;

typedef struct Buffer {
  NodeBuffer* left;
  NodeBuffer* right;
} Buffer;

typedef struct Constants {
  int L;

  c_float_t* p_ab;
  c_float_t* dw_p_ab;
  c_float_t* dv_p_ab;

  c_float_t* p_ij_cond;
  c_float_t* dw_p_ij_cond;
  c_float_t* dv_p_ij_cond;

  c_float_t* p_ji_cond;
  c_float_t* dw_p_ji_cond;
  c_float_t* dv_p_ji_cond;

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