#include "felsenstein.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>


void initialize_leaf(Node* leaf, Constants* consts) {

  uint8_t* msa = consts->msa;
  int L = consts->L;
  int i = consts->i;
  int j = consts->j;
  int a = msa[leaf->seq_id * L + i];
  int b = msa[leaf->seq_id * L + j];

  initialize_node(leaf);
  for(int ab = 0; ab < AA; ab++) {
    leaf->data->Ln_ab[ab] = -50;
  }
  leaf->data->Ln_ab[a*A + b] = 0;
  memset(leaf->data->dv_Ln_ab, 0, N_COL*AAA);
  for(int i = 0; i < N_COL*AAA; i++) {
    leaf->data->dv_Ln_ab[i] = -50;
  }
  for(int i = 0; i < AAAA; i++) {
    leaf->data->dw_Ln_ab[i] = -50;
  }
  memset(leaf->data->dv_Ln_ab_signs, 1, N_COL*AAA);
  memset(leaf->data->dw_Ln_ab_signs, 1, AAAA);
}

void initialize_buffer(NodeBuffer* buffer) {
  buffer->Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*A);
  buffer->Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*A);

  buffer->dv_Ln = (c_float_t*) malloc(sizeof(c_float_t)*N_COL*A);
  buffer->dv_Ln_signs = (int8_t*) malloc(sizeof(int8_t)*N_COL*A);
  buffer->dv_Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*N_COL*AA);
  buffer->dv_Ln_ia_signs = (int8_t*) malloc(sizeof(int8_t)*N_COL*AA);
  buffer->dv_Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*N_COL*AA);
  buffer->dv_Ln_jb_signs = (int8_t*) malloc(sizeof(int8_t)*N_COL*AA);

  buffer->dw_Ln = (c_float_t*) malloc(sizeof(c_float_t)*AA);
  buffer->dw_Ln_signs = (int8_t*) malloc(sizeof(int8_t)*AA);
  buffer->dw_Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*AAA);
  buffer->dw_Ln_ia_signs = (int8_t*) malloc(sizeof(int8_t)*AAA);
  buffer->dw_Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*AAA);
  buffer->dw_Ln_jb_signs = (int8_t*) malloc(sizeof(int8_t)*AAA);
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

  // d/dp p(Xm)
  c_float_t log_buffer_AA[AA];
  int8_t sign_buffer_2A[2*A];
  for(int a = 0; a < A; a++) {
    for(int b = 0; b < A; b++) {
      log_buffer_AA[a*A + b] = data->Ln_ab[a*A + b] + p_ab[a*A + b];
    }
  }
  buffer->Ln = logsumexpn(log_buffer_AA, AA);

  c_float_t log_buffer_2AA[2*AA];
  int8_t sign_buffer_2AA[2*AA];

  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for(int c_p = 0; c_p < A; c_p++) {
        for(int d_p = 0; d_p < A; d_p++) {
          int base_idx = 2*(c_p*A + d_p);
          log_buffer_2AA[base_idx] = data->dv_Ln_ab[l*AAA + c*AA + c_p*A + d_p] + p_ab[c_p*A + d_p];
          sign_buffer_2AA[base_idx] = data->dv_Ln_ab_signs[l*AAA + c*AA + c_p*A + d_p];
          log_buffer_2AA[base_idx + 1] = data->Ln_ab[c_p*A + d_p] + dv_p_ab[l*AAA + c*AA + c_p*A + d_p];
          sign_buffer_2AA[base_idx + 1] = dv_p_ab_signs[l*AAA + c*AA + c_p*A + d_p];
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2AA, sign_buffer_2AA, 2*AA);
      buffer->dv_Ln[l*A + c] = logsumexp_result.result;
      buffer->dv_Ln_signs[l*A + c] = logsumexp_result.sign;
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for(int c_p = 0; c_p < A; c_p++) {
        for(int d_p = 0; d_p < A; d_p++) {
          int base_idx = 2*(c_p*A + d_p);
          log_buffer_2AA[base_idx] = data->dw_Ln_ab[c*AAA + d*AA + c_p*A + d_p] + p_ab[c_p*A + d_p];
          sign_buffer_2AA[base_idx] = data->dw_Ln_ab_signs[c*AAA + d*AA + c_p*A + d_p];
          log_buffer_2AA[base_idx + 1] = data->Ln_ab[c_p*A + d_p] + dw_p_ab[c*AAA + d*AA + c_p*A + d_p];
          sign_buffer_2AA[base_idx + 1] = dw_p_ab_signs[c*AAA + d*AA + c_p*A + d_p];
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2AA, sign_buffer_2AA, 2*AA);
      buffer->dw_Ln[c*A + d] = logsumexp_result.result;
      buffer->dw_Ln_signs[c*A + d] = logsumexp_result.sign;
    }
  }

  // p(Xm|a, .)

  c_float_t log_buffer_A[A];
  for(int a = 0; a < A; a++) {
    for(int d = 0; d < A; d++) {
      log_buffer_A[d] = data->Ln_ab[a*A + d] + p_ji_cond[d*A + a];
    }
    buffer->Ln_ia[a] = logsumexpn(log_buffer_A, A);
  }

  int base_idx;
  c_float_t log_buffer_2A[2*A];
  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for (int a = 0; a < A; a++) {
        base_idx = 0;
        for (int d_p = 0; d_p < A; d_p++) {
          log_buffer_2A[base_idx] = data->dv_Ln_ab[l*AAA + c*AA + a*A + d_p] + p_ji_cond[d_p*A + a];
          sign_buffer_2A[base_idx] = data->dv_Ln_ab_signs[l*AAA + c*AA + a*A + d_p];
          base_idx++;
          if(l == 1) {
            log_buffer_2A[base_idx] = data->Ln_ab[a*A + d_p] + dv_p_ji_cond[l*AAA + c*AA + d_p*A + a];
            sign_buffer_2A[base_idx] = dv_p_ji_cond_signs[l*AAA + c*AA + d_p*A + a];
            base_idx++;
          }
        }
        SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A, sign_buffer_2A, base_idx);
        buffer->dv_Ln_ia[l*AA + c*A + a] = logsumexp_result.result;
        buffer->dv_Ln_ia_signs[l*AA + c*A + a] = logsumexp_result.sign;
      }
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for (int a = 0; a < A; a++) {
        for (int d_p = 0; d_p < A; d_p++) {
          int base_idx = 2*d_p;
          log_buffer_2A[base_idx] = data->dw_Ln_ab[c*AAA + d*AA + a*A + d_p] + p_ji_cond[d_p*A + a];
          sign_buffer_2A[base_idx] = data->dw_Ln_ab_signs[c*AAA + d*AA +a*A + d_p];
          log_buffer_2A[base_idx + 1] = data->Ln_ab[a*A + d_p] + dw_p_ji_cond[c*AAA + d*AA + d_p*A + a];
          sign_buffer_2A[base_idx + 1] = dw_p_ji_cond_signs[c*AAA + d*AA + d_p*A + a];
        }
        SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A, sign_buffer_2A, 2*A);
        buffer->dw_Ln_ia[c*AA + d*A + a] = logsumexp_result.result;
        buffer->dw_Ln_ia_signs[c*AA + d*A +a] = logsumexp_result.sign;
      }
    }
  }

  // d/dp p(Xm|.,b)
  for(int b = 0; b < A; b++) {
    for(int c = 0; c < A; c++) {
      log_buffer_A[c] = data->Ln_ab[c*A + b] + p_ij_cond[c*A + b];
    }
    buffer->Ln_jb[b] = logsumexpn(log_buffer_A, A);
  }

  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for (int b = 0; b < A; b++) {
        base_idx = 0;
        for (int c_p = 0; c_p < A; c_p++) {
          log_buffer_2A[base_idx] =  data->dv_Ln_ab[l*AAA + c*AA + c_p*A + b] + p_ij_cond[c_p*A + b];
          sign_buffer_2A[base_idx] = data->dv_Ln_ab_signs[l*AAA + c*AA + c_p*A + b];
          base_idx++;
          if( l == 0) {
            log_buffer_2A[base_idx] = data->Ln_ab[c_p*A + b] + dv_p_ij_cond[l*AAA + c*AA + c_p*A + b];
            sign_buffer_2A[base_idx] = dv_p_ij_cond_signs[l*AAA+ c*AA + c_p*A + b];
            base_idx++;
          }
        }
        SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A, sign_buffer_2A, base_idx);
        buffer->dv_Ln_jb[l*AA + c*A + b] = logsumexp_result.result;
        buffer->dv_Ln_jb_signs[l*AA + c*A + b] = logsumexp_result.sign;
      }
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for (int b = 0; b < A; b++) {
        for (int c_p = 0; c_p < A; c_p++) {
          int base_idx = 2*c_p;
          log_buffer_2A[base_idx] = data->dw_Ln_ab[c*AAA + d*AA + c_p*A + b] + p_ij_cond[c_p*A + b];
          sign_buffer_2A[base_idx] = data->dw_Ln_ab_signs[c*AAA + d*AA + c_p*A +b];
          log_buffer_2A[base_idx + 1] = data->Ln_ab[c_p*A + b] + dw_p_ij_cond[c*AAA + d*AA + c_p*A + b];
          sign_buffer_2A[base_idx + 1] = dw_p_ij_cond_signs[c*AAA + d*AA + c_p*A + b];
        }
        SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A, sign_buffer_2A, 2*A);
        buffer->dw_Ln_jb[c*AA + d*A + b] = logsumexp_result.result;
        buffer->dw_Ln_jb_signs[c*AA + d*A + b] = logsumexp_result.sign;
      }
    }
  }

}

void initialize_node(Node* node) {
  NodePrecomputation* data = malloc(sizeof(NodePrecomputation));
  data->Ln_ab = (c_float_t*) malloc(sizeof(c_float_t)*AA);
  data->dv_Ln_ab = (c_float_t*) malloc(sizeof(c_float_t)*N_COL*AAA);
  data->dv_Ln_ab_signs = (int8_t*) malloc(sizeof(int8_t)*N_COL*AAA);
  data->dw_Ln_ab = (c_float_t*) malloc(sizeof(c_float_t)*AAAA);
  data->dw_Ln_ab_signs = (int8_t*) malloc(sizeof(int8_t)*AAAA);
  node->data = data;
}

void deinitialize_node(Node* node) {
  NodePrecomputation* data = node->data;
  free(data->Ln_ab);
  free(data->dv_Ln_ab);
  free(data->dw_Ln_ab);
}

void compute_Ln_branch(Node* node, c_float_t phi, NodeBuffer* buffer, Constants* consts, c_float_t* L_ab, c_float_t* dv_L_ab, int8_t* dv_L_ab_signs, c_float_t* dw_L_ab, int8_t* dw_L_ab_signs, int a, int b){

  NodePrecomputation *child_data = node->data;
  NodeBuffer* child_buffer = buffer;

  c_float_t log_r = log(phi);
  c_float_t log_1mr = log(1 - phi);
  c_float_t mut2 = 2*log_1mr + child_buffer->Ln;
  c_float_t mut1 = log_r + log_1mr + logsumexp2(child_buffer->Ln_ia[a], child_buffer->Ln_jb[b]);
  c_float_t mut0 = 2*log_r + child_data->Ln_ab[a*A + b];
  *L_ab = logsumexp3(mut0, mut1, mut2);

  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      c_float_t ddv_mut2 = 2*log_1mr + child_buffer->dv_Ln[l*A +c];
      int8_t ddv_mut2_sign = child_buffer->dv_Ln_signs[l*A +c];

      c_float_t dv_Ln_ia = child_buffer->dv_Ln_ia[l*AA + c*A + a];
      int8_t dv_Ln_ia_sign = child_buffer->dv_Ln_ia_signs[l*AA + c*A + a];
      c_float_t dv_Ln_jb = child_buffer->dv_Ln_jb[l*AA + c*A + b];
      int8_t dv_Ln_jb_sign = child_buffer->dv_Ln_jb_signs[l*AA + c*A + b];
      SignedLogExp mut1_logsumexp = signed_logsumexp2(dv_Ln_ia, dv_Ln_ia_sign, dv_Ln_jb, dv_Ln_jb_sign);
      c_float_t ddv_mut1 = log_r + log_1mr + mut1_logsumexp.result;
      int8_t ddv_mut1_sign = mut1_logsumexp.sign;

      c_float_t ddv_mut0 = 2*log_r + child_data->dv_Ln_ab[l*AAA + c*AA + a*A + b];
      c_float_t ddv_mut0_sign = child_data->dv_Ln_ab_signs[l*AAA + c*AA + a*A + b];

      SignedLogExp mut_logsumexp = signed_logsumexp3(ddv_mut2, ddv_mut2_sign, ddv_mut1, ddv_mut1_sign, ddv_mut0, ddv_mut0_sign);
      dv_L_ab[l*A + c] = mut_logsumexp.result;
      dv_L_ab_signs[l*A + c] = mut_logsumexp.sign;
    }
  }
  for (int c = 0; c < A; c++) {
    for (int d = 0; d < A; d++) {
      c_float_t ddw_mut2 = 2*log_1mr + child_buffer->dw_Ln[c*A + d];
      int8_t ddw_mut2_sign = child_buffer->dw_Ln_signs[c*A + d];

      c_float_t dw_Ln_ia = child_buffer->dw_Ln_ia[c*AA + d*A + a];
      int8_t dw_Ln_ia_sign = child_buffer->dw_Ln_ia_signs[c*AA + d*A + a];
      c_float_t dw_Ln_jb = child_buffer->dw_Ln_jb[c*AA + d*A + b];
      int8_t dw_Ln_jb_sign = child_buffer->dw_Ln_jb_signs[c*AA + d*A + b];
      SignedLogExp mut1_logsumexp = signed_logsumexp2(dw_Ln_ia, dw_Ln_ia_sign, dw_Ln_jb, dw_Ln_jb_sign);
      c_float_t ddw_mut1 = log_r + log_1mr + mut1_logsumexp.result;
      int8_t ddw_mut1_sign = mut1_logsumexp.sign;

      c_float_t ddw_mut0 = 2*log_r + child_data->dw_Ln_ab[c*AAA + d*AA + a*A + b];
      c_float_t ddw_mut0_sign = child_data->dw_Ln_ab_signs[c*AAA + d*AA + a*A + b];

      SignedLogExp mut_logsumexp = signed_logsumexp3(ddw_mut2, ddw_mut2_sign, ddw_mut1, ddw_mut1_sign, ddw_mut0, ddw_mut0_sign);
      dw_L_ab[c*A + d] = mut_logsumexp.result;
      dw_L_ab_signs[c*A + d] = mut_logsumexp.sign;

    }
  }
}

void recurse_tree(Node* node, Constants* consts, Buffer* buf) {
  if (node->left == NULL && node->right == NULL) {
    // this is a leaf node
    initialize_leaf(node, consts);
    return;
  } else {
    recurse_tree(node->left, consts, buf);
    recurse_tree(node->right, consts, buf);
  }
  initialize_node(node);

  c_float_t* dv_left_Lab = (c_float_t *) malloc(sizeof(c_float_t)*N_COL*A);
  int8_t* dv_left_Lab_signs = (int8_t*) malloc(sizeof(int8_t)*N_COL*A);
  c_float_t* dv_right_Lab = (c_float_t *) malloc(sizeof(c_float_t)*N_COL*A);
  int8_t* dv_right_Lab_signs = (int8_t*) malloc(sizeof(c_float_t)*N_COL*A);

  c_float_t* dw_left_Lab = (c_float_t *) malloc(sizeof(c_float_t)*AA);
  int8_t* dw_left_Lab_signs = (int8_t*) malloc(sizeof(int8_t)*AA);
  c_float_t* dw_right_Lab = (c_float_t *) malloc(sizeof(c_float_t)*AA);
  int8_t* dw_right_Lab_signs = (int8_t*) malloc(sizeof(int8_t)*AA);


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

  for (int a = 0; a < A; a++) {
    for (int b = 0; b < A; b++) {
      c_float_t left_Lab = 0;
      c_float_t right_Lab = 0;

      if (node->left != NULL) {
        memset(dv_left_Lab, c_f0, N_COL*A);
        memset(dw_left_Lab, c_f0, AA);
        compute_Ln_branch(node->left, node->phi_left, buf->left, consts, &left_Lab, dv_left_Lab, dv_left_Lab_signs, dw_left_Lab, dw_left_Lab_signs, a, b);
      }
      if (node->right != NULL) {
        memset(dv_right_Lab, c_f0, N_COL*A);
        memset(dw_right_Lab, c_f0, AA);
        compute_Ln_branch(node->right, node->phi_right, buf->right, consts, &right_Lab, dv_right_Lab, dv_right_Lab_signs, dw_right_Lab, dw_right_Lab_signs, a,
                          b);
      }

      c_float_t Ln_ab = left_Lab + right_Lab;

      NodePrecomputation* node_data = node->data;
      node_data->Ln_ab[a*A + b] = Ln_ab;

      // combine derivatives for left and right children
      for(int l = 0; l < N_COL; l++){
        for(int c = 0; c < A; c++) {
          c_float_t left_deriv_term =  dv_left_Lab[l*A + c] + right_Lab;
          int8_t left_deriv_sign = dv_left_Lab_signs[l*A + c];
          c_float_t right_deriv_term = left_Lab + dv_right_Lab[l*A + c];
          int8_t right_deriv_sign = dv_right_Lab_signs[l*A + c];
          SignedLogExp logsumexp_result = signed_logsumexp2(left_deriv_term, left_deriv_sign, right_deriv_term, right_deriv_sign);
          node_data->dv_Ln_ab[l*AAA + c*AA + a*A + b] = logsumexp_result.result;
          node_data->dv_Ln_ab_signs[l*AAA + c*AA + a*A + b] = logsumexp_result.sign;
        }
      }

      for (int c = 0; c < A; c++) {
        for (int d = 0; d < A; d++) {
          c_float_t left_deriv_term = dw_left_Lab[c*A + d] + right_Lab;
          int8_t left_deriv_sign = dw_left_Lab_signs[c*A + d];
          c_float_t right_deriv_term = left_Lab + dw_right_Lab[c*A + d];
          int8_t right_deriv_sign = dw_right_Lab_signs[c*A + d];
          SignedLogExp logsumexp_result = signed_logsumexp2(left_deriv_term, left_deriv_sign, right_deriv_term, right_deriv_sign);
          node_data->dw_Ln_ab[c*AAA + d*AA + a*A + b] = logsumexp_result.result;
          node_data->dw_Ln_ab_signs[c*AAA + d*AA + a*A +b] = logsumexp_result.sign;
        }
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

  precalculate_constants(consts, x, x + N_COL*A);
  c_float_t* aa_freqs = consts->single_aa_frequencies;
  Node* root = consts->phylo_tree;

  recurse_tree(root, consts, buf);
  c_float_t buffer_AA_fx[AA];

  c_float_t fx = 0;
  for(int a = 0; a < A; a++) {
    for(int b = 0; b < A; b++) {
      buffer_AA_fx[a*A + b] =  root->data->Ln_ab[a*A + b] + aa_freqs[a] + aa_freqs[b];
    }
  }
  fx = logsumexpn(buffer_AA_fx, AA);

  c_float_t buffer_AA_grad[AA];
  memset(grad, c_f0, (N_COL*A + A*A)*sizeof(c_float_t));
  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for(int a = 0; a < A; a++) {
        for(int b = 0; b < A; b++) {
          buffer_AA_grad[a*A + b] = root->data->dv_Ln_ab[l*AAA + c*AA + a*A + b] + aa_freqs[a] + aa_freqs[b];
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(buffer_AA_grad, root->data->dv_Ln_ab_signs + l*AAA + c*AA, AA);
      grad[l*A + c] = logsumexp_result.sign * exp(logsumexp_result.result - fx);
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for(int a = 0; a < A; a++) {
        for(int b = 0; b < A; b++) {
          buffer_AA_grad[a*A + b] = root->data->dw_Ln_ab[c*AAA + d*AA + a*A + b] + aa_freqs[a] + aa_freqs[b];
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(buffer_AA_grad, root->data->dw_Ln_ab_signs + c*AAA + d*AA, AA);
      grad[N_COL*A + c*A + d] = logsumexp_result.sign * exp(logsumexp_result.result - fx);
    }
  }
  return fx;
}

void initialize_constants(Constants* consts) {
  consts->p_ab = (c_float_t*) malloc(sizeof(c_float_t) * AA);
  consts->dv_p_ab = (c_float_t*) malloc(sizeof(c_float_t) * N_COL*AAA);
  consts->dv_p_ab_signs = (int8_t*) malloc(sizeof(int8_t) * N_COL*AAA);
  consts->dw_p_ab = (c_float_t*) calloc(AAAA, sizeof(c_float_t));
  consts->dw_p_ab_signs = (int8_t*) malloc(sizeof(int8_t) * AAAA);
  consts->p_ij_cond = (c_float_t*) calloc(AA, sizeof(c_float_t));
  consts->dv_p_ij_cond = (c_float_t*) calloc(N_COL*AAA, sizeof(c_float_t));
  consts->dv_p_ij_cond_signs =  (int8_t*) malloc(sizeof(int8_t) * N_COL*AAA);
  consts->dw_p_ij_cond = (c_float_t*) calloc(AAAA, sizeof(c_float_t));
  consts->dw_p_ij_cond_signs =  (int8_t*) malloc(sizeof(int8_t) * AAAA);
  consts->p_ji_cond = (c_float_t*) calloc(AA, sizeof(c_float_t));
  consts->dv_p_ji_cond = (c_float_t*) calloc(N_COL*AAA, sizeof(c_float_t));
  consts->dv_p_ji_cond_signs =  (int8_t*) malloc(sizeof(int8_t) * N_COL*AAA);
  consts->dw_p_ji_cond = (c_float_t*) calloc(AAAA, sizeof(c_float_t));
  consts->dw_p_ji_cond_signs =  (int8_t*) malloc(sizeof(int8_t) * AAAA);
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
  // p_ab related precomputations
  c_float_t total_sum = 0;
  c_float_t *p_ab = consts->p_ab;
  memset(p_ab, c_f0, sizeof(c_float_t) * AA);
  for (int a = 0; a < A; a++) {
    for (int b = 0; b < A; b++) {
      p_ab[a * A + b] = exp(v[0 * A + a] + v[1 * A + b] + w[a * A + b]);
      total_sum += p_ab[a * A + b];
    }
  }
  for (int ab = 0; ab < A * A; ab++) {
    p_ab[ab] /= total_sum;
  }

  c_float_t pi_a[A] = {0};
  for (int a = 0; a < A; a++) {
    for (int b = 0; b < A; b++) {
      pi_a[a] += p_ab[a * A + b];
    }
  }
  c_float_t pj_b[A] = {0};
  for (int b = 0; b < A; b++) {
    for (int a = 0; a < A; a++) {
      pj_b[b] += p_ab[a * A + b];
    }
  }

  c_float_t *dv_p_ab = consts->dv_p_ab;
  memset(dv_p_ab, c_f0, sizeof(c_float_t) * N_COL * AAA);
  for (int c = 0; c < A; c++) {
    for (int a = 0; a < A; a++) {
      for (int b = 0; b < A; b++) {
        int ind = 0 * AAA + c * AA + a * A + b;
        dv_p_ab[ind] += (int) (a == c);
        dv_p_ab[ind] -= pi_a[c];
        dv_p_ab[ind] *= p_ab[a * A + b];
      }
    }
  }
  for (int d = 0; d < A; d++) {
    for (int a = 0; a < A; a++) {
      for (int b = 0; b < A; b++) {
        int ind = 1 * AAA + d * AA + a * A + b;
        dv_p_ab[ind] += (int) (b == d);
        dv_p_ab[ind] -= pj_b[d];
        dv_p_ab[ind] *= p_ab[a * A + b];
      }
    }
  }


  c_float_t *dw_p_ab = consts->dw_p_ab;
  memset(dw_p_ab, c_f0, sizeof(c_float_t) * AAAA);
  for (int c = 0; c < A; c++) {
    for (int d = 0; d < A; d++) {
      for (int a = 0; a < A; a++) {
        for (int b = 0; b < A; b++) {
          int ind = c * AAA + d * AA + a * A + b;
          dw_p_ab[ind] += (int) (a == c && b == d);
          dw_p_ab[ind] -= p_ab[c * A + d];
          dw_p_ab[ind] *= p_ab[a * A + b];
        }
      }
    }
  }

  // p(a,.|.,b)
  c_float_t *p_ij_cond = consts->p_ij_cond;
  memset(p_ij_cond, c_f0, sizeof(c_float_t) * AA);
  for (int b = 0; b < A; b++) {
    c_float_t ij_cond_sum = 0;
    for (int a = 0; a < A; a++) {
      c_float_t prob = exp(v[0 * A + a] + w[a * A + b]);
      p_ij_cond[a * A + b] = prob;
      ij_cond_sum += prob;
    }
    for (int a = 0; a < A; a++) {
      p_ij_cond[a * A + b] /= ij_cond_sum;
    }
  }

  c_float_t *dv_p_ij_cond = consts->dv_p_ij_cond;
  memset(dv_p_ij_cond, c_f0, sizeof(c_float_t) * N_COL * AAA);
  for (int c = 0; c < A; c++) {
    for (int a = 0; a < A; a++) {
      for (int b = 0; b < A; b++) {
        int ind = 0 * AAA + c * AA + b * A + a;
        dv_p_ij_cond[ind] += (int) (a == c);
        dv_p_ij_cond[ind] -= p_ij_cond[c * A + b];
        dv_p_ij_cond[ind] *= p_ij_cond[a * A + b];
      }
    }
  }

  c_float_t *dw_p_ij_cond = consts->dw_p_ij_cond;
  memset(dw_p_ij_cond, c_f0, sizeof(c_float_t) * AAAA);
  for (int c = 0; c < A; c++) {
    for (int d = 0; d < A; d++) {
      for (int a = 0; a < A; a++) {
        int b = d;
        int ind = c * AAA + d * AA + c * A + b;
        dw_p_ij_cond[ind] += (int) (a == c);
        dw_p_ij_cond[ind] -= p_ij_cond[c * A + d];
        dw_p_ij_cond[ind] *= p_ij_cond[a * A + b];
      }
    }
  }

  // p(.,b|a,.)
  c_float_t *p_ji_cond = consts->p_ji_cond;
  memset(p_ji_cond, c_f0, sizeof(c_float_t) * AA);
  for (int a = 0; a < A; a++) {
    c_float_t ji_cond_sum = 0;
    for (int b = 0; b < A; b++) {
      c_float_t prob = exp(v[1 * A + b] + w[a * A + b]);
      p_ji_cond[b * A + a] = prob;
      ji_cond_sum += prob;
    }
    for (int b = 0; b < A; b++) {
      p_ji_cond[b * A + a] /= ji_cond_sum;
    }
  }

  c_float_t *dv_p_ji_cond = consts->dv_p_ji_cond;
  memset(dv_p_ji_cond, c_f0, sizeof(c_float_t) * N_COL * AAA);
  for (int d = 0; d < A; d++) {
    for (int a = 0; a < A; a++) {
      for (int b = 0; b < A; b++) {
        int ind = 1 * AAA + d * AA + b * A + a;
        dv_p_ji_cond[ind] += (int) (b == d);
        dv_p_ji_cond[ind] -= p_ji_cond[d * A + a];
        dv_p_ji_cond[ind] *= p_ji_cond[b * A + a];
      }
    }
  }

  c_float_t *dw_p_ji_cond = consts->dw_p_ji_cond;
  memset(dw_p_ji_cond, c_f0, sizeof(c_float_t) * AAAA);
  for (int c = 0; c < A; c++) {
    for (int d = 0; d < A; d++) {
      for (int b = 0; b < A; b++) {
        int a = c;
        int ind = c * AAA + d * AA + b * A + a;
        dw_p_ji_cond[ind] += (int) (b == d);
        dw_p_ji_cond[ind] -= p_ji_cond[d * A + c];
        dw_p_ji_cond[ind] *= p_ji_cond[b * A + a];
      }
    }
  }

  for(int ab = 0; ab < AA; ab++) {
    p_ab[ab] = log(p_ab[ab]);
    p_ij_cond[ab] = log(p_ij_cond[ab]);
    p_ji_cond[ab] = log(p_ji_cond[ab]);
  }

  for(int ldab = 0; ldab < N_COL*AAA; ldab++) {
    int8_t sign;

    sign = dv_p_ab[ldab] >= 0 ? 1 : -1;
    dv_p_ab[ldab] = log(sign*dv_p_ab[ldab]);
    consts->dv_p_ab_signs[ldab] = sign;

    sign = dv_p_ij_cond[ldab] >= 0 ? 1 : -1;
    dv_p_ij_cond[ldab] = log(sign * dv_p_ij_cond[ldab]);
    consts->dv_p_ij_cond_signs[ldab] = sign;

    sign = dv_p_ji_cond[ldab] >= 0 ? 1 : -1;
    dv_p_ji_cond[ldab] = log(sign * dv_p_ji_cond[ldab]);
    consts->dv_p_ji_cond_signs[ldab] = sign;
  }

  for(int cdab = 0; cdab < AAAA; cdab++) {
    int8_t sign;

    sign = dw_p_ab[cdab] >= 0 ? 1 : -1;
    dw_p_ab[cdab] = log(sign*dw_p_ab[cdab]);
    consts->dw_p_ab_signs[cdab] = sign;

    sign = dw_p_ij_cond[cdab] >= 0 ? 1 : -1;
    dw_p_ij_cond[cdab] = log(sign * dw_p_ij_cond[cdab]);
    consts->dw_p_ij_cond_signs[cdab] = sign;

    sign = dw_p_ji_cond[cdab] >= 0 ? 1 : -1;
    dw_p_ji_cond[cdab] = log(sign * dw_p_ji_cond[cdab]);
    consts->dw_p_ji_cond_signs[cdab] = sign;
  }
}