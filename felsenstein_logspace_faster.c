#include "felsenstein_faster.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>


void initialize_leaf(Node* leaf, Constants* consts) {

  int A_a = consts->A_i;
  int A_b = consts->A_b;
  int AA_ab = consts->AA_ij;
  int A_a_p_A_b = consts->A_i_p_A_j;

  uint8_t* msa = consts->msa;
  int L = consts->L;
  int i = consts->i;
  int j = consts->j;
  int a = msa[leaf->seq_id * L + i];
  int b = msa[leaf->seq_id * L + j];

  initialize_node(leaf, consts);
  for(int ab = 0; ab < AA_ab; ab++) {
    leaf->data->Ln_ab[ab] = log0;
  }
  leaf->data->Ln_ab[a*A_b + b] = 0;
  memset(leaf->data->dv_Ln_ab, 0, A_a_p_A_b*AA_ab);
  for(int i = 0; i < A_a_p_A_b*AA_ab; i++) {
    leaf->data->dv_Ln_ab[i] = log0;
  }
  for(int i = 0; i < AA_ab*AA_ab; i++) {
    leaf->data->dw_Ln_ab[i] = log0;
  }
  memset(leaf->data->dv_Ln_ab_signs, 1, A_a_p_A_b*AA_ab);
  memset(leaf->data->dw_Ln_ab_signs, 1, AA_ab*AA_ab);
}

void initialize_buffer(NodeBuffer* buffer, Constants* consts) {

  int A_a = consts->A_i;
  int A_b = consts->A_b;
  int A_a_p_A_b = consts->A_i_p_A_j;
  int AA_ab = consts->AA_ij;

  buffer->Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*A_a);
  buffer->Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*A_b);

  buffer->dv_Ln = (c_float_t*) malloc(sizeof(c_float_t)*A_a_p_A_b);
  buffer->dv_Ln_signs = (int8_t*) malloc(sizeof(int8_t)*A_a_p_A_b);
  buffer->dv_Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*A_a_p_A_b*A_a);
  buffer->dv_Ln_ia_signs = (int8_t*) malloc(sizeof(int8_t)*A_a_p_A_b*A_a);
  buffer->dv_Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*A_a_p_A_b*A_b);
  buffer->dv_Ln_jb_signs = (int8_t*) malloc(sizeof(int8_t)*A_a_p_A_b*A_b);

  buffer->dw_Ln = (c_float_t*) malloc(sizeof(c_float_t)*AA_ab);
  buffer->dw_Ln_signs = (int8_t*) malloc(sizeof(int8_t)*AA_ab);
  buffer->dw_Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*AA_ab*A_a);
  buffer->dw_Ln_ia_signs = (int8_t*) malloc(sizeof(int8_t)*AA_ab*A_a);
  buffer->dw_Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*AA_ab*A_b);
  buffer->dw_Ln_jb_signs = (int8_t*) malloc(sizeof(int8_t)*AA_ab*A_b);
}

void deinitialize_buffer(NodeBuffer* buffer) {
  free(buffer->Ln_ia);
  free(buffer->Ln_jb);

  free(buffer->dv_Ln);
  free(buffer->dv_Ln_signs);
  free(buffer->dv_Ln_ia);
  free(buffer->dv_Ln_ia_signs);
  free(buffer->dv_Ln_jb);
  free(buffer->dv_Ln_jb_signs);

  free(buffer->dw_Ln);
  free(buffer->dw_Ln_signs);
  free(buffer->dw_Ln_ia);
  free(buffer->dw_Ln_ia_signs);
  free(buffer->dw_Ln_jb);
  free(buffer->dw_Ln_jb_signs);
}

void precompute_buffer(NodeBuffer* buffer, NodePrecomputation* data, Constants* consts){

  int A_a = consts->A_i;
  int A_b = consts->A_b;
  int AA_ab = consts->AA_ij;
  int A_a_p_A_b = consts->A_i_p_A_j;

  c_float_t *p_ab = consts->p_ab;
  c_float_t *dv_p_ab = consts->dv_p_ab;
  int8_t *dv_p_ab_signs = consts->dv_p_ab_signs;
  c_float_t *dw_p_ab = consts->dw_p_ab;
  int8_t *dw_p_ab_signs = consts->dw_p_ab_signs;

  c_float_t *p_ij_cond = consts->p_ij_cond;
  c_float_t *dv_p_ij_cond = consts->dv_p_ij_cond;
  int8_t *dv_p_ij_cond_signs = consts->dv_p_ij_cond_signs;
  c_float_t *dw_p_ij_cond = consts->dw_p_ij_cond;
  int8_t *dw_p_ij_cond_signs = consts->dw_p_ij_cond_signs;

  c_float_t *p_ji_cond = consts->p_ji_cond;
  c_float_t *dv_p_ji_cond = consts->dv_p_ji_cond;
  int8_t *dv_p_ji_cond_signs = consts->dv_p_ji_cond_signs;
  c_float_t *dw_p_ji_cond = consts->dw_p_ji_cond;
  int8_t *dw_p_ji_cond_signs = consts->dw_p_ji_cond_signs;

  // Nulling out buffer
  buffer->Ln = 0;

  /*
  memset(buffer->Ln_ia, 0, sizeof(c_float_t)*A);
  memset(buffer->Ln_jb, 0, sizeof(c_float_t)*A);

  memset(buffer->dv_Ln, 0, sizeof(c_float_t)*N_COL*A);
  memset(buffer->dv_Ln_ia, 0, sizeof(c_float_t)*N_COL*AA);
  memset(buffer->dv_Ln_jb, 0, sizeof(c_float_t)*N_COL*AA);

  memset(buffer->dw_Ln, 0, sizeof(c_float_t)*AA);
  memset(buffer->dw_Ln_ia, 0, sizeof(c_float_t)*AAA);
  memset(buffer->dw_Ln_jb, 0, sizeof(c_float_t)*AAA);
  */

  c_float_t log_buffer_A_b[A_b];
  c_float_t log_buffer_A_a[A_a];

  c_float_t log_buffer_2A_a[2*A_a];
  int8_t sign_buffer_2A_a[2*A_a];
  c_float_t log_buffer_2A_b[2*A_b];
  int8_t sign_buffer_2A_b[2*A_b];

  c_float_t log_buffer_AA[AA_ab];
  c_float_t log_buffer_2AA[2*AA_ab];
  int8_t sign_buffer_2AA[2*AA_ab];


  // d/dp p(Xm)
  for(int a = 0; a < A_a; a++) {
    for(int b = 0; b < A_b; b++) {
      log_buffer_AA[a*A_b + b] = data->Ln_ab[a*A_b + b] + p_ab[a*A_b + b];
    }
  }
  buffer->Ln = logsumexpn(log_buffer_AA, AA_ab);

  for(int lc = 0; lc < A_a_p_A_b; lc++) {
    for(int c_p = 0; c_p < A_a; c_p++) {
      for(int d_p = 0; d_p < A_b; d_p++) {
        int base_idx = 2*(c_p*A_b + d_p);
        log_buffer_2AA[base_idx] = data->dv_Ln_ab[lc*AA_ab + c_p*A_b + d_p] + p_ab[c_p*A_b + d_p];
        sign_buffer_2AA[base_idx] = data->dv_Ln_ab_signs[lc*AA_ab + c_p*A_b + d_p];
        log_buffer_2AA[base_idx + 1] = data->Ln_ab[c_p*A_b + d_p] + dv_p_ab[lc*AA_ab + c_p*A_b + d_p];
        sign_buffer_2AA[base_idx + 1] = dv_p_ab_signs[lc*AA_ab + c_p*A_b + d_p];
      }
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2AA, sign_buffer_2AA, 2*AA_ab);
    buffer->dv_Ln[lc] = logsumexp_result.result;
    buffer->dv_Ln_signs[lc] = logsumexp_result.sign;
  }

  for(int cd = 0; cd < AA_ab; cd++) {
    for(int c_p = 0; c_p < A_a; c_p++) {
      for(int d_p = 0; d_p < A_b; d_p++) {
        int base_idx = 2*(c_p*A_b + d_p);
        log_buffer_2AA[base_idx] = data->dw_Ln_ab[cd*AA_ab + c_p*A_b + d_p] + p_ab[c_p*A_b + d_p];
        sign_buffer_2AA[base_idx] = data->dw_Ln_ab_signs[cd*AA_ab + c_p*A_b + d_p];
        log_buffer_2AA[base_idx + 1] = data->Ln_ab[c_p*A_b + d_p] + dw_p_ab[cd*AA_ab + c_p*A_b + d_p];
        sign_buffer_2AA[base_idx + 1] = dw_p_ab_signs[cd*AA_ab + c_p*A_b + d_p];
      }
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2AA, sign_buffer_2AA, 2*AA_ab);
    buffer->dw_Ln[cd] = logsumexp_result.result;
    buffer->dw_Ln_signs[cd] = logsumexp_result.sign;
  }

  // p(Xm|a, .)
  for(int a = 0; a < A_a; a++) {
    for(int d = 0; d < A_b; d++) {
      log_buffer_A_b[d] = data->Ln_ab[a*A_b + d] + p_ji_cond[d*A_a + a];
    }
    buffer->Ln_ia[a] = logsumexpn(log_buffer_A_b, A_b);
  }

  int base_idx;
  for(int lc = 0; lc < A_a_p_A_b; lc++) {
    for (int a = 0; a < A_a; a++) {
      base_idx = 0;
      for (int d_p = 0; d_p < A_b; d_p++) {
        log_buffer_2A_b[base_idx] = data->dv_Ln_ab[lc*AA_ab + a*A_b + d_p] + p_ji_cond[d_p*A_a + a];
        sign_buffer_2A_b[base_idx] = data->dv_Ln_ab_signs[lc*AA_ab + a*A_b + d_p];
        base_idx++;
        if(lc >= A_a) {
          log_buffer_2A_b[base_idx] = data->Ln_ab[a*A_b + d_p] + dv_p_ji_cond[lc*AA_ab + d_p*A_a + a];
          sign_buffer_2A_b[base_idx] = dv_p_ji_cond_signs[lc*AA_ab + d_p*A_a + a];
          base_idx++;
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_b, sign_buffer_2A_b, base_idx);
      buffer->dv_Ln_ia[lc*A_a + a] = logsumexp_result.result;
      buffer->dv_Ln_ia_signs[lc*A_a + a] = logsumexp_result.sign;
    }
  }


  for(int cd = 0; cd < AA_ab; cd++) {
    for (int a = 0; a < A_a; a++) {
      for (int d_p = 0; d_p < A_b; d_p++) {
        int base_idx = 2*d_p;
        log_buffer_2A_b[base_idx] = data->dw_Ln_ab[cd*AA_ab + a*A_b + d_p] + p_ji_cond[d_p*A_a + a];
        sign_buffer_2A_b[base_idx] = data->dw_Ln_ab_signs[cd*AA_ab + a*A_b + d_p];
        log_buffer_2A_b[base_idx + 1] = data->Ln_ab[a*A_b + d_p] + dw_p_ji_cond[cd*AA_ab + d_p*A_a + a];
        sign_buffer_2A_b[base_idx + 1] = dw_p_ji_cond_signs[cd*AA_ab + d_p*A_a + a];
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_b, sign_buffer_2A_b, 2*A_b);
      buffer->dw_Ln_ia[cd*A_a + a] = logsumexp_result.result;
      buffer->dw_Ln_ia_signs[cd*A_a + a] = logsumexp_result.sign;
    }
  }

  // d/dp p(Xm|.,b)
  for(int b = 0; b < A_b; b++) {
    for(int c = 0; c < A_a; c++) {
      log_buffer_A_a[c] = data->Ln_ab[c*A_b + b] + p_ij_cond[c*A_b + b];
    }
    buffer->Ln_jb[b] = logsumexpn(log_buffer_A_a, A_a);
  }

  for(int lc = 0; lc < A_a_p_A_b; lc++) {
    for (int b = 0; b < A_b; b++) {
      base_idx = 0;
      for (int c_p = 0; c_p < A_a; c_p++) {
        log_buffer_2A_a[base_idx] =  data->dv_Ln_ab[lc*AA_ab + c_p*A_b + b] + p_ij_cond[c_p*A_b + b];
        sign_buffer_2A_a[base_idx] = data->dv_Ln_ab_signs[lc*AA_ab + c_p*A_b + b];
        base_idx++;
        if(lc < A_a) {
          log_buffer_2A_a[base_idx] = data->Ln_ab[c_p*A_b + b] + dv_p_ij_cond[lc*AA_ab + c_p*A_b + b];
          sign_buffer_2A_a[base_idx] = dv_p_ij_cond_signs[lc*AA_ab + c_p*A_b + b];
          base_idx++;
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_a, sign_buffer_2A_a, base_idx);
      buffer->dv_Ln_jb[lc*A_b + b] = logsumexp_result.result;
      buffer->dv_Ln_jb_signs[lc*A_b + b] = logsumexp_result.sign;
    }
  }
  for(int cd = 0; cd < AA_ab; cd++) {
    for (int b = 0; b < A_b; b++) {
      for (int c_p = 0; c_p < A_a; c_p++) {
        int base_idx = 2*c_p;
        log_buffer_2A_a[base_idx] = data->dw_Ln_ab[cd*AA_ab + c_p*A_b + b] + p_ij_cond[c_p*A_b + b];
        sign_buffer_2A_a[base_idx] = data->dw_Ln_ab_signs[cd*AA_ab + c_p*A_b +b];
        log_buffer_2A_a[base_idx + 1] = data->Ln_ab[c_p*A_b + b] + dw_p_ij_cond[cd*AA_ab + c_p*A_b + b];
        sign_buffer_2A_a[base_idx + 1] = dw_p_ij_cond_signs[cd*AA_ab + c_p*A_b + b];
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_a, sign_buffer_2A_a, 2*A_a);
      buffer->dw_Ln_jb[cd*A_b + b] = logsumexp_result.result;
      buffer->dw_Ln_jb_signs[cd*A_b + b] = logsumexp_result.sign;
    }
  }

}

void initialize_node(Node* node, Constants* consts) {

  int AA_ab = consts->AA_ij;
  int A_a_p_A_b = consts->A_i_p_A_j;

  NodePrecomputation* data = malloc(sizeof(NodePrecomputation));
  data->Ln_ab = (c_float_t*) malloc(sizeof(c_float_t)*AA_ab);
  data->dv_Ln_ab = (c_float_t*) malloc(sizeof(c_float_t)*A_a_p_A_b*AA_ab);
  data->dv_Ln_ab_signs = (int8_t*) malloc(sizeof(int8_t)*A_a_p_A_b*AA_ab);
  data->dw_Ln_ab = (c_float_t*) malloc(sizeof(c_float_t)*AA_ab*AA_ab);
  data->dw_Ln_ab_signs = (int8_t*) malloc(sizeof(int8_t)*AA_ab*AA_ab);
  node->data = data;
}

void deinitialize_node(Node* node) {
  NodePrecomputation* data = node->data;
  free(data->Ln_ab);
  free(data->dv_Ln_ab);
  free(data->dv_Ln_ab_signs);
  free(data->dw_Ln_ab);
  free(data->dw_Ln_ab_signs);
  free(node->data);
}

void compute_Ln_branch(Node* node, c_float_t phi, NodeBuffer* buffer, Constants* consts, c_float_t* L_ab, c_float_t* dv_L_ab, int8_t* dv_L_ab_signs, c_float_t* dw_L_ab, int8_t* dw_L_ab_signs, int a, int b){

  int A_a = consts->A_i;
  int A_b = consts->A_b;
  int AA_ab = consts->AA_ij;
  int A_a_p_A_b = consts->A_i_p_A_j;

  NodePrecomputation *child_data = node->data;
  NodeBuffer* child_buffer = buffer;

  c_float_t log_r = log2(phi);
  c_float_t log_1mr = log2(1 - phi);
  c_float_t mut2 = 2*log_1mr + child_buffer->Ln;
  c_float_t mut1 = log_r + log_1mr + logsumexp2(child_buffer->Ln_ia[a], child_buffer->Ln_jb[b]);
  c_float_t mut0 = 2*log_r + child_data->Ln_ab[a*A_b + b];
  *L_ab = logsumexp3(mut0, mut1, mut2);

  for(int lc = 0; lc < A_a_p_A_b; lc++) {
    c_float_t ddv_mut2 = 2*log_1mr + child_buffer->dv_Ln[lc];
    int8_t ddv_mut2_sign = child_buffer->dv_Ln_signs[lc];

    c_float_t dv_Ln_ia = child_buffer->dv_Ln_ia[lc*A_a + a];
    int8_t dv_Ln_ia_sign = child_buffer->dv_Ln_ia_signs[lc*A_a + a];
    c_float_t dv_Ln_jb = child_buffer->dv_Ln_jb[lc*A_b + b];
    int8_t dv_Ln_jb_sign = child_buffer->dv_Ln_jb_signs[lc*A_b + b];
    SignedLogExp mut1_logsumexp = signed_logsumexp2(dv_Ln_ia, dv_Ln_ia_sign, dv_Ln_jb, dv_Ln_jb_sign);
    c_float_t ddv_mut1 = log_r + log_1mr + mut1_logsumexp.result;
    int8_t ddv_mut1_sign = mut1_logsumexp.sign;

    c_float_t ddv_mut0 = 2*log_r + child_data->dv_Ln_ab[lc*AA_ab + a*A_b + b];
    c_float_t ddv_mut0_sign = child_data->dv_Ln_ab_signs[lc*AA_ab + a*A_b + b];

    SignedLogExp mut_logsumexp = signed_logsumexp3(ddv_mut2, ddv_mut2_sign, ddv_mut1, ddv_mut1_sign, ddv_mut0, ddv_mut0_sign);
    dv_L_ab[lc] = mut_logsumexp.result;
    dv_L_ab_signs[lc] = mut_logsumexp.sign;
  }

  for (int cd = 0; cd < AA_ab; cd++) {
    c_float_t ddw_mut2 = 2*log_1mr + child_buffer->dw_Ln[cd];
    int8_t ddw_mut2_sign = child_buffer->dw_Ln_signs[cd];

    c_float_t dw_Ln_ia = child_buffer->dw_Ln_ia[cd*A_a + a];
    int8_t dw_Ln_ia_sign = child_buffer->dw_Ln_ia_signs[cd*A_a + a];
    c_float_t dw_Ln_jb = child_buffer->dw_Ln_jb[cd*A_b + b];
    int8_t dw_Ln_jb_sign = child_buffer->dw_Ln_jb_signs[cd*A_b + b];
    SignedLogExp mut1_logsumexp = signed_logsumexp2(dw_Ln_ia, dw_Ln_ia_sign, dw_Ln_jb, dw_Ln_jb_sign);
    c_float_t ddw_mut1 = log_r + log_1mr + mut1_logsumexp.result;
    int8_t ddw_mut1_sign = mut1_logsumexp.sign;

    c_float_t ddw_mut0 = 2*log_r + child_data->dw_Ln_ab[cd*AA_ab + a*A_b + b];
    c_float_t ddw_mut0_sign = child_data->dw_Ln_ab_signs[cd*AA_ab + a*A_b + b];

    SignedLogExp mut_logsumexp = signed_logsumexp3(ddw_mut2, ddw_mut2_sign, ddw_mut1, ddw_mut1_sign, ddw_mut0, ddw_mut0_sign);
    dw_L_ab[cd] = mut_logsumexp.result;
    dw_L_ab_signs[cd] = mut_logsumexp.sign;
  }
}

void recurse_tree(Node* node, Constants* consts, Buffer* buf) {

  int A_a = consts->A_i;
  int A_b = consts->A_b;
  int AA_ab = consts->AA_ij;
  int A_a_p_A_b = consts->A_i_p_A_j;

  if (node->left == NULL && node->right == NULL) {
    // this is a leaf node
    initialize_leaf(node, consts);
    return;
  } else {
    recurse_tree(node->left, consts, buf);
    recurse_tree(node->right, consts, buf);
  }
  initialize_node(node, consts);

  c_float_t dv_left_Lab[A_a_p_A_b];
  int8_t dv_left_Lab_signs[A_a_p_A_b];
  c_float_t dv_right_Lab[A_a_p_A_b];
  int8_t dv_right_Lab_signs[A_a_p_A_b];

  c_float_t dw_left_Lab[AA_ab];
  int8_t dw_left_Lab_signs[AA_ab];
  c_float_t dw_right_Lab[AA_ab];
  int8_t dw_right_Lab_signs[AA_ab];


  // precalculate aggregated values
  if(node->left != NULL) {
    NodePrecomputation *left_data = node->left->data;
    NodeBuffer* left_buf = buf->left;
    precompute_buffer(left_buf, left_data, consts);
  }
  if(node->right != NULL) {
    NodePrecomputation* right_data = node->right->data;
    NodeBuffer* right_buf = buf->right;
    precompute_buffer(right_buf, right_data, consts);
  }

  for (int a = 0; a < A_a; a++) {
    for (int b = 0; b < A_b; b++) {
      c_float_t left_Lab = 0;
      c_float_t right_Lab = 0;

      if (node->left != NULL) {
        memset(dv_left_Lab, c_f0, A_a_p_A_b * sizeof(c_float_t));
        memset(dw_left_Lab, c_f0, AA_ab * sizeof(c_float_t));
        compute_Ln_branch(node->left, node->phi_left, buf->left, consts, &left_Lab, dv_left_Lab, dv_left_Lab_signs,
                          dw_left_Lab, dw_left_Lab_signs, a, b);
      }
      if (node->right != NULL) {
        memset(dv_right_Lab, c_f0, A_a_p_A_b * sizeof(c_float_t));
        memset(dw_right_Lab, c_f0, AA_ab * sizeof(c_float_t));
        compute_Ln_branch(node->right, node->phi_right, buf->right, consts, &right_Lab, dv_right_Lab,
                          dv_right_Lab_signs, dw_right_Lab, dw_right_Lab_signs, a,
                          b);
      }

      c_float_t Ln_ab = left_Lab + right_Lab;

      NodePrecomputation *node_data = node->data;
      node_data->Ln_ab[a * A_b + b] = Ln_ab;

      // combine derivatives for left and right children
      for (int lc = 0; lc < A_a_p_A_b; lc++) {
        c_float_t left_deriv_term = dv_left_Lab[lc] + right_Lab;
        int8_t left_deriv_sign = dv_left_Lab_signs[lc];
        c_float_t right_deriv_term = left_Lab + dv_right_Lab[lc];
        int8_t right_deriv_sign = dv_right_Lab_signs[lc];
        SignedLogExp logsumexp_result = signed_logsumexp2(left_deriv_term, left_deriv_sign, right_deriv_term,
                                                          right_deriv_sign);
        node_data->dv_Ln_ab[lc * AA_ab + a * A_b + b] = logsumexp_result.result;
        node_data->dv_Ln_ab_signs[lc * AA_ab + a * A_b + b] = logsumexp_result.sign;
      }

      for (int cd = 0; cd < AA_ab; cd++) {
        c_float_t left_deriv_term = dw_left_Lab[cd] + right_Lab;
        int8_t left_deriv_sign = dw_left_Lab_signs[cd];
        c_float_t right_deriv_term = left_Lab + dw_right_Lab[cd];
        int8_t right_deriv_sign = dw_right_Lab_signs[cd];
        SignedLogExp logsumexp_result = signed_logsumexp2(left_deriv_term, left_deriv_sign, right_deriv_term,
                                                          right_deriv_sign);
        node_data->dw_Ln_ab[cd * AA_ab + a * A_b + b] = logsumexp_result.result;
        node_data->dw_Ln_ab_signs[cd * AA_ab + a * A_b + b] = logsumexp_result.sign;
      }
    }
  }
  if(node->left != NULL) {
    deinitialize_node(node->left);
  }
  if(node->right != NULL) {
    deinitialize_node(node->right);
  }
}

c_float_t calculate_fx_grad(c_float_t*x, c_float_t* grad, Constants* consts, Buffer* buf) {

  int A_a = consts->A_i;
  int A_b = consts->A_b;
  int AA_ab = consts->AA_ij;
  int A_a_p_A_b = consts->A_i_p_A_j;

  precalculate_constants(consts, x, x + A_a_p_A_b);
  Node* root = consts->phylo_tree;

  recurse_tree(root, consts, buf);
  c_float_t buffer_AA_fx[AA_ab];

  c_float_t fx = 0;
  for(int a = 0; a < A_a; a++) {
    for(int b = 0; b < A_b; b++) {
      buffer_AA_fx[a*A_b + b] =  root->data->Ln_ab[a*A_b + b] + consts->p_ab[a*A_b + b];
    }
  }
  fx = logsumexpn(buffer_AA_fx, AA_ab);

  memset(grad, c_f0, (A_a_p_A_b + AA_ab)*sizeof(c_float_t));
  c_float_t buffer_2AA_grad[2*AA_ab];
  int8_t buffer_2AA_signs[2*AA_ab];

  for(int lc = 0; lc < A_a_p_A_b; lc++) {
    for(int ab = 0; ab < AA_ab; ab++) {
      int base_idx = 2*ab;
      buffer_2AA_grad[base_idx] = root->data->dv_Ln_ab[lc*AA_ab + ab] + consts->p_ab[ab];
      buffer_2AA_signs[base_idx] = root->data->dv_Ln_ab_signs[lc*AA_ab + ab];
      buffer_2AA_grad[base_idx + 1] = root->data->Ln_ab[ab] + consts->dv_p_ab[lc*AA_ab + ab];
      buffer_2AA_signs[base_idx + 1] = consts->dv_p_ab_signs[lc*AA_ab + ab];
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(buffer_2AA_grad, buffer_2AA_signs, 2*AA_ab);
    grad[lc] = logsumexp_result.sign * pow(2, logsumexp_result.result - fx);
  }
  for(int cd = 0; cd < AA_ab; cd++) {
    for(int ab = 0; ab < AA_ab; ab++) {
      int base_idx = 2*ab;
      buffer_2AA_grad[base_idx] = root->data->dw_Ln_ab[cd*AA_ab + ab] + consts->p_ab[ab];
      buffer_2AA_signs[base_idx] = root->data->dw_Ln_ab_signs[cd*AA_ab + ab];
      buffer_2AA_grad[base_idx + 1] = root->data->Ln_ab[ab] + consts->dw_p_ab[cd*AA_ab + ab];
      buffer_2AA_signs[base_idx + 1] = consts->dw_p_ab_signs[cd*AA_ab + ab];
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(buffer_2AA_grad, buffer_2AA_signs, 2*AA_ab);
    grad[A_a_p_A_b + cd] = logsumexp_result.sign * pow(2, logsumexp_result.result - fx);
  }
  deinitialize_node(root);
  return fx;
}

void initialize_constants(Constants* consts) {
  int AA_ab = consts->AA_ij;
  int A_a_p_A_b = consts->A_i_p_A_j;

  consts->p_ab = (c_float_t*) malloc(sizeof(c_float_t) * AA_ab);
  consts->dv_p_ab = (c_float_t*) malloc(sizeof(c_float_t) * A_a_p_A_b*AA_ab);
  consts->dv_p_ab_signs = (int8_t*) malloc(sizeof(int8_t) *  A_a_p_A_b*AA_ab);
  consts->dw_p_ab = (c_float_t*) calloc(AA_ab*AA_ab, sizeof(c_float_t));
  consts->dw_p_ab_signs = (int8_t*) malloc(sizeof(int8_t) * AA_ab*AA_ab);
  consts->p_ij_cond = (c_float_t*) calloc(AA_ab, sizeof(c_float_t));
  consts->dv_p_ij_cond = (c_float_t*) calloc( A_a_p_A_b*AA_ab, sizeof(c_float_t));
  consts->dv_p_ij_cond_signs =  (int8_t*) malloc(sizeof(int8_t) * A_a_p_A_b*AA_ab);
  consts->dw_p_ij_cond = (c_float_t*) calloc(AA_ab*AA_ab, sizeof(c_float_t));
  consts->dw_p_ij_cond_signs =  (int8_t*) malloc(sizeof(int8_t) * AA_ab*AA_ab);
  consts->p_ji_cond = (c_float_t*) calloc(AA_ab, sizeof(c_float_t));
  consts->dv_p_ji_cond = (c_float_t*) calloc( A_a_p_A_b*AA_ab, sizeof(c_float_t));
  consts->dv_p_ji_cond_signs =  (int8_t*) malloc(sizeof(int8_t) *  A_a_p_A_b*AA_ab);
  consts->dw_p_ji_cond = (c_float_t*) calloc(AA_ab*AA_ab, sizeof(c_float_t));
  consts->dw_p_ji_cond_signs =  (int8_t*) malloc(sizeof(int8_t) * AA_ab*AA_ab);
}

void deinitialize_constants(Constants* consts) {
  free(consts->p_ab);
  free(consts->dv_p_ab);
  free(consts->dv_p_ab_signs);
  free(consts->dw_p_ab);
  free(consts->dw_p_ab_signs);
  free(consts->p_ij_cond);
  free(consts->dv_p_ij_cond);
  free(consts->dv_p_ij_cond_signs);
  free(consts->dw_p_ij_cond);
  free(consts->dw_p_ij_cond_signs);
  free(consts->p_ji_cond);
  free(consts->dv_p_ji_cond);
  free(consts->dv_p_ji_cond_signs);
  free(consts->dw_p_ji_cond);
  free(consts->dw_p_ji_cond_signs);
}

void precalculate_constants(Constants* consts, c_float_t* v, c_float_t* w) {

  int A_a = consts->A_i;
  int A_b = consts->A_b;
  int A_a_p_A_b = consts->A_i_p_A_j;
  int AA_ab = consts->AA_ij;

  // p_ab related precomputations
  c_float_t total_sum = 0;
  c_float_t *p_ab = consts->p_ab;
  initialize_array(p_ab, log0, AA_ab);
  for (int a = 0; a < A_a; a++) {
    for (int b = 0; b < A_b; b++) {
      p_ab[a*A_b + b] = v[a] + v[A_a + b] + w[a*A_b + b];
      total_sum += p_ab[a * A_b + b];
    }
  }
  c_float_t normalization = logsumexpn(p_ab, AA_ab);
  for (int ab = 0; ab < AA_ab; ab++) {
    p_ab[ab] -= normalization;
  }

  c_float_t pi_a[A_a];
  for (int a = 0; a < A_a; a++) {
    pi_a[a] = logsumexpn(p_ab + a*A_b, A_b);
  }
  c_float_t pj_b[A_b];
  c_float_t a_tmp[A_a];
  for (int b = 0; b < A_b; b++) {
    for (int a = 0; a < A_a; a++) {
      a_tmp[a] = p_ab[a*A_b + b];
    }
    pj_b[b] = logsumexpn(a_tmp, A_a);
  }

  c_float_t *dv_p_ab = consts->dv_p_ab;
  initialize_array(dv_p_ab, log0,  A_a_p_A_b*AA_ab);
  for (int c = 0; c < A_a; c++) {
    for (int a = 0; a < A_a; a++) {
      for (int b = 0; b < A_b; b++) {
        int ind = c * AA_ab + a*A_b + b;
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? 0: log0, 1,  pi_a[c], -1);
        dv_p_ab[ind] = logsumexp_result.result + p_ab[a*A_b+ b];
        consts->dv_p_ab_signs[ind] = logsumexp_result.sign;
      }
    }
  }
  for (int d = 0; d < A_b; d++) {
    for (int a = 0; a < A_a; a++) {
      for (int b = 0; b < A_b; b++) {
        int ind = (A_a + d) * AA_ab + a*A_b + b;
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? 0: log0, 1,  pj_b[d], -1);
        dv_p_ab[ind] = logsumexp_result.result + p_ab[a * A_b + b];
        consts->dv_p_ab_signs[ind] = logsumexp_result.sign;
      }
    }
  }


  c_float_t *dw_p_ab = consts->dw_p_ab;
  initialize_array(dw_p_ab, log0,  AA_ab*AA_ab);
  for (int c = 0; c < A_a; c++) {
    for (int d = 0; d < A_b; d++) {
      for (int a = 0; a < A_a; a++) {
        for (int b = 0; b < A_b; b++) {
          int ind = c * AA_ab*A_b + d * AA_ab + a*A_b + b;
          SignedLogExp logsumexp_result = signed_logsumexp2((a == c && b == d) ? 0: log0, 1,  p_ab[c*A_b + d], -1);
          dw_p_ab[ind] = logsumexp_result.result + p_ab[a * A_b + b];
          consts->dw_p_ab_signs[ind] = logsumexp_result.sign;
        }
      }
    }
  }

  // p(a,.|.,b)
  c_float_t *p_ij_cond = consts->p_ij_cond;
  initialize_array(p_ij_cond, log0, AA_ab);
  c_float_t tmp_prob[(A_a > A_b) ? A_a : A_b];
  for (int b = 0; b < A_b; b++) {
    for (int a = 0; a < A_a; a++) {
      c_float_t log_prob = v[a] + w[a * A_b + b];
      tmp_prob[a] = log_prob;
      p_ij_cond[a*A_b + b] = log_prob;
    }
    c_float_t norm = logsumexpn(tmp_prob, A_a);
    for (int a = 0; a < A_a; a++) {
      p_ij_cond[a * A_b + b] -= norm;
    }
  }

  c_float_t *dv_p_ij_cond = consts->dv_p_ij_cond;
  initialize_array(dv_p_ij_cond, log0, A_a_p_A_b*AA_ab);
  for (int c = 0; c < A_a; c++) {
    for (int a = 0; a < A_a; a++) {
      for (int b = 0; b < A_b; b++) {
        int ind = c * AA_ab + a*A_b + b;
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? 0: log0, 1,  p_ij_cond[c*A_b + b], -1);
        dv_p_ij_cond[ind] = logsumexp_result.result + p_ij_cond[a*A_b + b];
        consts->dv_p_ij_cond_signs[ind] = logsumexp_result.sign;
      }
    }
  }

  c_float_t *dw_p_ij_cond = consts->dw_p_ij_cond;
  initialize_array(dw_p_ij_cond, log0,AA_ab*AA_ab);
  for (int c = 0; c < A_a; c++) {
    for (int d = 0; d < A_b; d++) {
      for (int a = 0; a < A_a; a++) {
        int b = d;
        int ind = c * AA_ab*A_b + d*AA_ab + a*A_b + b;
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? 0: log0, 1,  p_ij_cond[c*A_b + d], -1);
        dw_p_ij_cond[ind] = logsumexp_result.result + p_ij_cond[a*A_b + b];
        consts->dw_p_ij_cond_signs[ind] = logsumexp_result.sign;
      }
    }
  }

  // p(.,b|a,.)
  c_float_t *p_ji_cond = consts->p_ji_cond;
  initialize_array(p_ji_cond, log0, AA_ab);
  for (int a = 0; a < A_a; a++) {
    for (int b = 0; b < A_b; b++) {
      c_float_t prob = v[A_a + b] + w[a*A_b + b];
      p_ji_cond[b*A_a + a] = prob;
      tmp_prob[b] = prob;
    }
    c_float_t normalization = logsumexpn(tmp_prob, A_b);
    for (int b = 0; b < A_b; b++) {
      p_ji_cond[b*A_a + a] -= normalization;
    }
  }

  c_float_t *dv_p_ji_cond = consts->dv_p_ji_cond;
  initialize_array(dv_p_ji_cond, log0, A_a_p_A_b*AA_ab);
  for (int d = 0; d < A_b; d++) {
    for (int a = 0; a < A_a; a++) {
      for (int b = 0; b < A_b; b++) {
        int ind = A_a*AA_ab + d*AA_ab + b*A_a + a;
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? 0: log0, 1, p_ji_cond[d*A_a + a], -1);
        dv_p_ji_cond[ind] = logsumexp_result.result + p_ji_cond[b*A_a + a];
        consts->dv_p_ji_cond_signs[ind] = logsumexp_result.sign;
      }
    }
  }

  c_float_t *dw_p_ji_cond = consts->dw_p_ji_cond;
  initialize_array(dw_p_ji_cond, log0, AA_ab*AA_ab);
  for (int c = 0; c < A_a; c++) {
    for (int d = 0; d < A_b; d++) {
      for (int b = 0; b < A_b; b++) {
        int a = c;
        int ind = c*AA_ab*A_b + d*AA_ab + b*A_a + a;
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? 0: log0, 1, p_ji_cond[d*A_a + c], -1);
        dw_p_ji_cond[ind] = logsumexp_result.result + p_ji_cond[b*A_a + a];
        consts->dw_p_ji_cond_signs[ind] = logsumexp_result.sign;
      }
    }
  }
}
