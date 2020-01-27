#include "felsenstein_simdstructure.h"
#include <stdlib.h>
#include <math.h>

#include "simd.h"
#include "simd_functions.h"

#define pad_float(N)  (N + (VECSIZE_FLOAT-1))/VECSIZE_FLOAT*VECSIZE_FLOAT

void initialize_leaf(Node* leaf, Constants* consts) {

  int A_j = consts->A_j;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij_padded = consts->AA_ij_padded;

  uint8_t* msa = consts->msa;
  int L = consts->L;
  int i = consts->i;
  int j = consts->j;
  int a = msa[leaf->seq_id * L + i];
  int b = msa[leaf->seq_id * L + j];

  initialize_node(leaf, consts);
  for(int ab = 0; ab < AA_ij; ab++) {
    leaf->data->Ln_ab[ab] = log0;
  }
  leaf->data->Ln_ab[a * A_j + b] = 0;
  initialize_array(leaf->data->dv_Ln_ab, c_f0, A_i_p_A_j * AA_ij);
  for(int i = 0; i < A_i_p_A_j * AA_ij; i++) {
    leaf->data->dv_Ln_ab[i] = log0;
  }
  for(int i = 0; i < AA_ij * AA_ij_padded; i++) {
    leaf->data->dw_Ln_ab[i] = log0;
  }
  initialize_array(leaf->data->dv_Ln_ab_signs, c_f1, A_i_p_A_j * AA_ij);
  initialize_array(leaf->data->dw_Ln_ab_signs, c_f1, AA_ij * AA_ij_padded);
}

void initialize_nodebuffer(NodeBuffer* buffer, Constants* consts) {

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij_padded = consts->AA_ij_padded;

  buffer->Ln_ia = (c_float_t*) malloc(sizeof(c_float_t) * A_i);
  buffer->Ln_jb = (c_float_t*) malloc(sizeof(c_float_t) * A_j);

  buffer->dv_Ln = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j);
  buffer->dv_Ln_signs = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j);
  buffer->dv_Ln_ia = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * A_i);
  buffer->dv_Ln_ia_signs = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * A_i);
  buffer->dv_Ln_jb = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * A_j);
  buffer->dv_Ln_jb_signs = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * A_j);

  buffer->dw_Ln = malloc_simd_float(sizeof(c_float_t)*AA_ij_padded);
  buffer->dw_Ln_signs = malloc_simd_float(sizeof(c_float_t)*AA_ij_padded);
  buffer->dw_Ln_ia = malloc_simd_float(sizeof(c_float_t) * A_i * AA_ij_padded);
  buffer->dw_Ln_ia_signs = malloc_simd_float(sizeof(c_float_t) * A_i * AA_ij_padded);
  buffer->dw_Ln_jb = malloc_simd_float(sizeof(c_float_t) * A_j * AA_ij_padded);
  buffer->dw_Ln_jb_signs = malloc_simd_float(sizeof(c_float_t) * A_j * AA_ij_padded);

  buffer->dw_mut0_buffer = malloc_simd_float(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_mut1_buffer = malloc_simd_float(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_mut1_sign_buffer = malloc_simd_float(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_mut2_buffer = malloc_simd_float(AA_ij_padded*sizeof(c_float_t));

}

void initialize_buffer(Buffer* buffer, Constants* consts) {
  int AA_ij_padded = consts->AA_ij_padded;
  buffer->left = malloc(sizeof(NodeBuffer));
  initialize_nodebuffer(buffer->left, consts);
  buffer->right = malloc(sizeof(NodeBuffer));
  initialize_nodebuffer(buffer->right, consts);

  buffer->dw_left_Lab =  malloc_simd_float(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_right_Lab = malloc_simd_float(AA_ij_padded*sizeof(c_float_t));
}

void deinitialize_buffer(Buffer* buffer) {
  deinitialize_nodebuffer(buffer->left);
  deinitialize_nodebuffer(buffer->right);

  free(buffer->dw_left_Lab);
  free(buffer->dw_right_Lab);
}

void deinitialize_nodebuffer(NodeBuffer* buffer) {
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

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij_padded = consts->AA_ij_padded;

  c_float_t *dv_p_ab = consts->dv_p_ab;
  c_float_t *dv_p_ab_signs = consts->dv_p_ab_signs;

  c_float_t (*p_ij_cond)[A_j] = (c_float_t (*)[A_j]) consts->p_ij_cond;
  c_float_t *dv_p_ij_cond = consts->dv_p_ij_cond;
  c_float_t *dv_p_ij_cond_signs = consts->dv_p_ij_cond_signs;

  c_float_t (*p_ji_cond)[A_j] =  (c_float_t (*)[A_j]) consts->p_ji_cond;
  c_float_t *dv_p_ji_cond = consts->dv_p_ji_cond;
  c_float_t *dv_p_ji_cond_signs = consts->dv_p_ji_cond_signs;


  c_float_t (* dw_Ln_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) data->dw_Ln_ab;
  c_float_t (* dw_Ln_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) data->dw_Ln_ab_signs;

  c_float_t (* Ln_ab)[A_j] = (c_float_t (*)[A_j]) data->Ln_ab;
  c_float_t (* dw_p_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ab;
  c_float_t (* dw_p_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ab_signs;
  c_float_t (* p_ab)[A_j] = (c_float_t (*)[A_j]) consts->p_ab;
  c_float_t (* dw_p_ij_cond)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ij_cond;
  c_float_t (* dw_p_ij_cond_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ij_cond_signs;
  c_float_t (* dw_p_ji_cond)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ji_cond;
  c_float_t (* dw_p_ji_cond_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ji_cond_signs;

  c_float_t (*dw_Ln_ia)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) buffer->dw_Ln_ia;
  c_float_t (*dw_Ln_ia_signs)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) buffer->dw_Ln_ia_signs;
  c_float_t (*dw_Ln_jb)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) buffer->dw_Ln_jb;
  c_float_t (*dw_Ln_jb_signs)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) buffer->dw_Ln_jb_signs;


  // Nulling out buffer
  buffer->Ln = 0;

  c_float_t log_buffer_A_j[A_j];
  c_float_t log_buffer_A_i[A_i];

  c_float_t log_buffer_2A_i[2 * A_i];
  c_float_t sign_buffer_2A_i[2 * A_i];
  c_float_t log_buffer_2A_j[2 * A_j];
  c_float_t sign_buffer_2A_j[2 * A_j];

  c_float_t log_buffer_AA[AA_ij];
  c_float_t log_buffer_2AA[2 * AA_ij];
  c_float_t sign_buffer_2AA[2 * AA_ij];


  // d/dp p(Xm)
  for(int a = 0; a < A_i; a++) {
    for(int b = 0; b < A_j; b++) {
      log_buffer_AA[a*A_j + b] = data->Ln_ab[a*A_j + b] + p_ab[a][b];
    }
  }
  buffer->Ln = logsumexpn(log_buffer_AA, AA_ij);

  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for(int c_p = 0; c_p < A_i; c_p++) {
      for(int d_p = 0; d_p < A_j; d_p++) {
        int base_idx = 2*(c_p*A_j + d_p);
        log_buffer_2AA[base_idx] = data->dv_Ln_ab[lc * AA_ij + c_p * A_j + d_p] + p_ab[c_p][d_p];
        sign_buffer_2AA[base_idx] = data->dv_Ln_ab_signs[lc * AA_ij + c_p * A_j + d_p];
        log_buffer_2AA[base_idx + 1] = data->Ln_ab[c_p*A_j + d_p] + dv_p_ab[lc * AA_ij + c_p * A_j + d_p];
        sign_buffer_2AA[base_idx + 1] = dv_p_ab_signs[lc * AA_ij + c_p * A_j + d_p];
      }
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2AA, sign_buffer_2AA, 2 * AA_ij);
    buffer->dv_Ln[lc] = logsumexp_result.result;
    buffer->dv_Ln_signs[lc] = logsumexp_result.sign;
  }

  logsumexp_matrix_ax01(A_i, A_j, AA_ij_padded,
    buffer->dw_Ln, buffer->dw_Ln_signs,
    dw_Ln_ab, dw_Ln_ab_signs, p_ab,
    dw_p_ab, dw_p_ab_signs, Ln_ab
    );
  /*
  for(int cd = 0; cd < AA_ij; cd++) {
    for(int c_p = 0; c_p < A_i; c_p++) {
      for(int d_p = 0; d_p < A_j; d_p++) {
        int base_idx = 2*(c_p*A_j + d_p);
        log_buffer_2AA[base_idx] = dw_Ln_ab[c_p][d_p][cd]+ p_ab[c_p][d_p];
        sign_buffer_2AA[base_idx] = dw_Ln_ab_signs[c_p][d_p][cd];
        log_buffer_2AA[base_idx + 1] = Ln_ab[c_p][d_p] + dw_p_ab[c_p][d_p][cd];
        sign_buffer_2AA[base_idx + 1] = dw_p_ab_signs[c_p][d_p][cd];
      }
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2AA, sign_buffer_2AA, 2*AA_ij);
    buffer->dw_Ln[cd] = logsumexp_result.result;
    buffer->dw_Ln_signs[cd] = logsumexp_result.sign;
  }
   */

  // p(Xm|a, .)
  for(int a = 0; a < A_i; a++) {
    for(int d = 0; d < A_j; d++) {
      log_buffer_A_j[d] = data->Ln_ab[a * A_j + d] + p_ji_cond[a][d];
    }
    buffer->Ln_ia[a] = logsumexpn(log_buffer_A_j, A_j);
  }

  int base_idx;
  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for (int a = 0; a < A_i; a++) {
      base_idx = 0;
      for (int d_p = 0; d_p < A_j; d_p++) {
        log_buffer_2A_j[base_idx] = data->dv_Ln_ab[lc * AA_ij + a * A_j + d_p] + p_ji_cond[a][d_p];
        sign_buffer_2A_j[base_idx] = data->dv_Ln_ab_signs[lc * AA_ij + a * A_j + d_p];
        base_idx++;
        if(lc >= A_i) {
          log_buffer_2A_j[base_idx] = data->Ln_ab[a * A_j + d_p] + dv_p_ji_cond[lc * AA_ij + d_p * A_i + a];
          sign_buffer_2A_j[base_idx] = dv_p_ji_cond_signs[lc * AA_ij + d_p * A_i + a];
          base_idx++;
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_j, sign_buffer_2A_j, base_idx);
      buffer->dv_Ln_ia[lc*A_i + a] = logsumexp_result.result;
      buffer->dv_Ln_ia_signs[lc*A_i + a] = logsumexp_result.sign;
    }
  }

  logsumexp_matrix_ax1(A_i, A_j, AA_ij_padded,
    dw_Ln_ia, dw_Ln_ia_signs,
    dw_Ln_ab, dw_Ln_ab_signs, p_ji_cond,
    dw_p_ji_cond, dw_p_ji_cond_signs, Ln_ab
  );

  /*
  for(int cd = 0; cd < AA_ij; cd++) {
    for (int a = 0; a < A_i; a++) {
      for (int d_p = 0; d_p < A_j; d_p++) {
        int base_idx = 2*d_p;
        log_buffer_2A_j[base_idx] = dw_Ln_ab[a][d_p][cd] + p_ji_cond[a][d_p];
        sign_buffer_2A_j[base_idx] = dw_Ln_ab_signs[a][d_p][cd];
        log_buffer_2A_j[base_idx + 1] = Ln_ab[a][d_p] + dw_p_ji_cond[a][d_p][cd];
        sign_buffer_2A_j[base_idx + 1] = dw_p_ji_cond_signs[a][d_p][cd];
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_j, sign_buffer_2A_j, 2*A_j);
      dw_Ln_ia[a][cd] = logsumexp_result.result;
      dw_Ln_ia_signs[a][cd] = logsumexp_result.sign;
    }
  }
   */

  // d/dp p(Xm|.,b)
  for(int b = 0; b < A_j; b++) {
    for(int c = 0; c < A_i; c++) {
      log_buffer_A_i[c] = data->Ln_ab[c * A_j + b] + p_ij_cond[c][b];
    }
    buffer->Ln_jb[b] = logsumexpn(log_buffer_A_i, A_i);
  }

  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for (int b = 0; b < A_j; b++) {
      base_idx = 0;
      for (int c_p = 0; c_p < A_i; c_p++) {
        log_buffer_2A_i[base_idx] = data->dv_Ln_ab[lc * AA_ij + c_p * A_j + b] + p_ij_cond[c_p][b];
        sign_buffer_2A_i[base_idx] = data->dv_Ln_ab_signs[lc * AA_ij + c_p * A_j + b];
        base_idx++;
        if(lc < A_i) {
          log_buffer_2A_i[base_idx] = data->Ln_ab[c_p * A_j + b] + dv_p_ij_cond[lc * AA_ij + c_p * A_j + b];
          sign_buffer_2A_i[base_idx] = dv_p_ij_cond_signs[lc * AA_ij + c_p * A_j + b];
          base_idx++;
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_i, sign_buffer_2A_i, base_idx);
      buffer->dv_Ln_jb[lc*A_j + b] = logsumexp_result.result;
      buffer->dv_Ln_jb_signs[lc*A_j + b] = logsumexp_result.sign;
    }
  }


  logsumexp_matrix_ax0(A_i, A_j, AA_ij_padded,
                       dw_Ln_jb, dw_Ln_jb_signs,
                       dw_Ln_ab, dw_Ln_ab_signs, p_ij_cond,
                       dw_p_ij_cond, dw_p_ij_cond_signs, Ln_ab
  );

  /*
  for(int cd = 0; cd < AA_ij; cd++) {
    for (int b = 0; b < A_j; b++) {
      for (int c_p = 0; c_p < A_i; c_p++) {
        int base_idx = 2*c_p;
        log_buffer_2A_i[base_idx] = dw_Ln_ab[c_p][b][cd] + p_ij_cond[c_p*A_j + b];
        sign_buffer_2A_i[base_idx] = dw_Ln_ab_signs[c_p][b][cd];
        log_buffer_2A_i[base_idx + 1] = data->Ln_ab[c_p*A_j + b] + dw_p_ij_cond[c_p][b][cd];
        sign_buffer_2A_i[base_idx + 1] = dw_p_ij_cond_signs[c_p][b][cd];
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_i, sign_buffer_2A_i, 2*A_i);
      dw_Ln_jb[b][cd] = logsumexp_result.result;
      dw_Ln_jb_signs[b][cd] = logsumexp_result.sign;
    }
  }
   */

}

void initialize_node(Node* node, Constants* consts) {

  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij_padded = consts->AA_ij_padded;

  NodePrecomputation* data = malloc(sizeof(NodePrecomputation));
  data->Ln_ab = (c_float_t*) malloc(sizeof(c_float_t) * AA_ij);
  data->dv_Ln_ab = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * AA_ij);
  data->dv_Ln_ab_signs = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * AA_ij);
  data->dw_Ln_ab = (c_float_t*) malloc(sizeof(c_float_t) * AA_ij * AA_ij_padded);
  data->dw_Ln_ab_signs = (c_float_t*) malloc(sizeof(c_float_t) * AA_ij * AA_ij_padded);
  node->data = data;

  node->log_1mp_left = log2(1 - node->phi_left);
  node->log_p_left = log2(node->phi_left);

  node->log_1mp_right = log2(1 - node->phi_right);
  node->log_p_right = log2(node->phi_right);
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

void compute_Ln_branch(Node* node, c_float_t log_r, c_float_t log_1mr, NodeBuffer* buffer, Constants* consts, c_float_t* L_ab, c_float_t* dv_L_ab, c_float_t* dv_L_ab_signs, c_float_t* dw_L_ab, c_float_t* dw_L_ab_signs, int a, int b){

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij_padded = consts->AA_ij_padded;

  NodePrecomputation *child_data = node->data;
  NodeBuffer* child_buffer = buffer;

  c_float_t Ln = child_buffer->Ln;
  c_float_t* Ln_ia = child_buffer->Ln_ia;
  c_float_t* Ln_jb = child_buffer->Ln_jb;
  c_float_t (*Ln_ab)[A_j] = (c_float_t (*)[A_j]) child_data->Ln_ab;

  c_float_t mut2 = 2*log_1mr + Ln;
  c_float_t mut1 = log_r + log_1mr + logsumexp2(Ln_ia[a], Ln_jb[b]);
  c_float_t mut0 = 2*log_r + Ln_ab[a][b];
  *L_ab = logsumexp3(mut0, mut1, mut2);


  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    c_float_t ddv_mut2 = 2*log_1mr + child_buffer->dv_Ln[lc];
    c_float_t ddv_mut2_sign = child_buffer->dv_Ln_signs[lc];

    c_float_t dv_Ln_ia = child_buffer->dv_Ln_ia[lc * A_i + a];
    c_float_t dv_Ln_ia_sign = child_buffer->dv_Ln_ia_signs[lc * A_i + a];
    c_float_t dv_Ln_jb = child_buffer->dv_Ln_jb[lc * A_j + b];
    c_float_t dv_Ln_jb_sign = child_buffer->dv_Ln_jb_signs[lc * A_j + b];
    SignedLogExp mut1_logsumexp = signed_logsumexp2(dv_Ln_ia, dv_Ln_ia_sign, dv_Ln_jb, dv_Ln_jb_sign);
    c_float_t ddv_mut1 = log_r + log_1mr + mut1_logsumexp.result;
    c_float_t ddv_mut1_sign = mut1_logsumexp.sign;

    c_float_t ddv_mut0 = 2*log_r + child_data->dv_Ln_ab[lc * AA_ij + a * A_j + b];
    c_float_t ddv_mut0_sign = child_data->dv_Ln_ab_signs[lc * AA_ij + a * A_j + b];

    SignedLogExp mut_logsumexp = signed_logsumexp3(ddv_mut2, ddv_mut2_sign, ddv_mut1, ddv_mut1_sign, ddv_mut0, ddv_mut0_sign);
    dv_L_ab[lc] = mut_logsumexp.result;
    dv_L_ab_signs[lc] = mut_logsumexp.sign;
  }

  c_float_t (*dw_Ln_ia)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_ia;
  c_float_t (*dw_Ln_ia_signs)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_ia_signs;
  c_float_t (*dw_Ln_jb)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_jb;
  c_float_t (*dw_Ln_jb_signs)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_jb_signs;
  c_float_t (*dw_Ln_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) child_data->dw_Ln_ab;
  c_float_t (*dw_Ln_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) child_data->dw_Ln_ab_signs;

  add_constant(buffer->dw_mut2_buffer,  child_buffer->dw_Ln, 2*log_1mr, AA_ij_padded);
  signedlogsumexp2_array(buffer->dw_mut1_buffer, buffer->dw_mut1_sign_buffer,
    dw_Ln_ia[a], dw_Ln_ia_signs[a],
    dw_Ln_jb[b],  dw_Ln_jb_signs[b],
    AA_ij_padded
    );
  add_constant(buffer->dw_mut1_buffer, buffer->dw_mut1_buffer, log_r + log_1mr,  AA_ij_padded);
  add_constant(buffer->dw_mut0_buffer, dw_Ln_ab[a][b], 2*log_r, AA_ij_padded);
  signedlogsumexp3_array(dw_L_ab, dw_L_ab_signs,
    buffer->dw_mut2_buffer, child_buffer->dw_Ln_signs,
    buffer->dw_mut1_buffer, buffer->dw_mut1_sign_buffer,
    buffer->dw_mut0_buffer, dw_Ln_ab_signs[a][b],
    AA_ij_padded
  );

  /*

  for (int cd = 0; cd < AA_ij; cd++) {

    c_float_t ddw_mut2 = 2*log_1mr + child_buffer->dw_Ln[cd];
    c_float_t ddw_mut2_sign = child_buffer->dw_Ln_signs[cd];

    c_float_t dw_Ln_ia_val = dw_Ln_ia[a][cd];
    c_float_t dw_Ln_ia_sign_val = dw_Ln_ia_signs[a][cd];
    c_float_t dw_Ln_jb_val = dw_Ln_jb[b][cd];
    c_float_t dw_Ln_jb_sign_val = dw_Ln_jb_signs[b][cd];
    SignedLogExp mut1_logsumexp = signed_logsumexp2(dw_Ln_ia_val, dw_Ln_ia_sign_val, dw_Ln_jb_val, dw_Ln_jb_sign_val);
    c_float_t ddw_mut1 = log_r + log_1mr + mut1_logsumexp.result;
    c_float_t ddw_mut1_sign = mut1_logsumexp.sign;

    c_float_t ddw_mut0 = 2*log_r + dw_Ln_ab[a][b][cd];
    c_float_t ddw_mut0_sign = dw_Ln_ab_signs[a][b][cd];

    SignedLogExp mut_logsumexp = signed_logsumexp3(ddw_mut2, ddw_mut2_sign, ddw_mut1, ddw_mut1_sign, ddw_mut0, ddw_mut0_sign);
    dw_L_ab[cd] = mut_logsumexp.result;
    dw_L_ab_signs[cd] = mut_logsumexp.sign;
  }
   */
}

void recurse_tree(Node* node, Constants* consts, Buffer* buf) {

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij_padded = consts->AA_ij_padded;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;

  if (node->left == NULL && node->right == NULL) {
    // this is a leaf node
    initialize_leaf(node, consts);
    return;
  } else {
    recurse_tree(node->left, consts, buf);
    recurse_tree(node->right, consts, buf);
  }
  initialize_node(node, consts);

  c_float_t dv_left_Lab[A_i_p_A_j];
  c_float_t dv_left_Lab_signs[A_i_p_A_j];
  c_float_t dv_right_Lab[A_i_p_A_j];
  c_float_t dv_right_Lab_signs[A_i_p_A_j];

  c_float_t dw_left_Lab[AA_ij_padded];
  c_float_t dw_left_Lab_signs[AA_ij_padded];
  c_float_t dw_right_Lab[AA_ij_padded];
  c_float_t dw_right_Lab_signs[AA_ij_padded];


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

  for (int a = 0; a < A_i; a++) {
    for (int b = 0; b < A_j; b++) {
      c_float_t left_Lab = 0;
      c_float_t right_Lab = 0;

      if (node->left != NULL) {
        initialize_array(dv_left_Lab, c_f0, A_i_p_A_j);
        initialize_array(dw_left_Lab, c_f0, AA_ij_padded);
        compute_Ln_branch(node->left, node->log_p_left, node->log_1mp_left, buf->left, consts, &left_Lab, dv_left_Lab, dv_left_Lab_signs,
                          dw_left_Lab, dw_left_Lab_signs, a, b);
      }
      if (node->right != NULL) {
        initialize_array(dv_right_Lab, c_f0, A_i_p_A_j);
        initialize_array(dw_right_Lab, c_f0, AA_ij_padded);
        compute_Ln_branch(node->right, node->log_p_right, node->log_1mp_right, buf->right, consts, &right_Lab, dv_right_Lab,
                          dv_right_Lab_signs, dw_right_Lab, dw_right_Lab_signs, a, b);
      }

      c_float_t Ln_ab = left_Lab + right_Lab;

      NodePrecomputation *node_data = node->data;
      node_data->Ln_ab[a * A_j + b] = Ln_ab;

      // combine derivatives for left and right children
      for (int lc = 0; lc < A_i_p_A_j; lc++) {
        c_float_t left_deriv_term = dv_left_Lab[lc] + right_Lab;
        c_float_t left_deriv_sign = dv_left_Lab_signs[lc];
        c_float_t right_deriv_term = left_Lab + dv_right_Lab[lc];
        c_float_t right_deriv_sign = dv_right_Lab_signs[lc];
        SignedLogExp logsumexp_result = signed_logsumexp2(left_deriv_term, left_deriv_sign, right_deriv_term,
                                                          right_deriv_sign);
        node_data->dv_Ln_ab[lc * AA_ij + a * A_j + b] = logsumexp_result.result;
        node_data->dv_Ln_ab_signs[lc * AA_ij + a * A_j + b] = logsumexp_result.sign;
      }

      c_float_t (*dw_Ln_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) node_data->dw_Ln_ab;
      c_float_t (*dw_Ln_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) node_data->dw_Ln_ab_signs;

      //TODO AVX implementation

      add_constant(buf->dw_left_Lab, dw_left_Lab, right_Lab, AA_ij_padded);
      add_constant(buf->dw_right_Lab, dw_right_Lab, left_Lab, AA_ij_padded);
      signedlogsumexp2_array(dw_Ln_ab[a][b], dw_Ln_ab_signs[a][b],
        buf->dw_left_Lab, dw_left_Lab_signs,
        buf->dw_right_Lab, dw_right_Lab_signs,
        AA_ij_padded
        );
      /*
      for (int cd = 0; cd < AA_ij; cd++) {
        c_float_t left_deriv_term = dw_left_Lab[cd] + right_Lab;
        c_float_t left_deriv_sign = dw_left_Lab_signs[cd];
        c_float_t right_deriv_term = left_Lab + dw_right_Lab[cd];
        c_float_t right_deriv_sign = dw_right_Lab_signs[cd];
        SignedLogExp logsumexp_result = signed_logsumexp2(left_deriv_term, left_deriv_sign, right_deriv_term,
                                                          right_deriv_sign);
        dw_Ln_ab[a][b][cd] = logsumexp_result.result;
        dw_Ln_ab_signs[a][b][cd] = logsumexp_result.sign;
      }
       */
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

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij_padded = consts->AA_ij_padded;

  precalculate_constants(consts, x, x + A_i_p_A_j);
  Node* root = consts->phylo_tree;

  recurse_tree(root, consts, buf);
  c_float_t buffer_AA_fx[AA_ij];

  c_float_t fx = 0;
  for(int a = 0; a < A_i; a++) {
    for(int b = 0; b < A_j; b++) {
      buffer_AA_fx[a * A_j + b] = root->data->Ln_ab[a * A_j + b] + consts->p_ab[a * A_j + b];
    }
  }
  fx = logsumexpn(buffer_AA_fx, AA_ij);

  initialize_array(grad, c_f0, A_i_p_A_j + AA_ij);
  c_float_t buffer_2AA_grad[2 * AA_ij];
  c_float_t buffer_2AA_signs[2 * AA_ij];

  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for(int ab = 0; ab < AA_ij; ab++) {
      int base_idx = 2*ab;
      buffer_2AA_grad[base_idx] = root->data->dv_Ln_ab[lc * AA_ij + ab] + consts->p_ab[ab];
      buffer_2AA_signs[base_idx] = root->data->dv_Ln_ab_signs[lc * AA_ij + ab];
      buffer_2AA_grad[base_idx + 1] = root->data->Ln_ab[ab] + consts->dv_p_ab[lc * AA_ij + ab];
      buffer_2AA_signs[base_idx + 1] = consts->dv_p_ab_signs[lc * AA_ij + ab];
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(buffer_2AA_grad, buffer_2AA_signs, 2 * AA_ij);
    grad[lc] = logsumexp_result.sign * pow(2, logsumexp_result.result - fx);
  }

  c_float_t (*dw_Ln_ab)[AA_ij_padded] = (c_float_t(*)[AA_ij_padded]) root->data->dw_Ln_ab;
  c_float_t (*dw_Ln_ab_signs)[AA_ij_padded] = (c_float_t(*)[AA_ij_padded]) root->data->dw_Ln_ab_signs;
  c_float_t (*dw_p_ab)[AA_ij_padded] = (c_float_t(*)[AA_ij_padded]) consts->dw_p_ab;
  c_float_t (*dw_p_ab_signs)[AA_ij_padded] = (c_float_t(*)[AA_ij_padded]) consts->dw_p_ab_signs;


  for(int cd = 0; cd < AA_ij; cd++) {
    for(int ab = 0; ab < AA_ij; ab++) {
      int base_idx = 2*ab;
      buffer_2AA_grad[base_idx] = dw_Ln_ab[ab][cd] + consts->p_ab[ab];
      buffer_2AA_signs[base_idx] = dw_Ln_ab_signs[ab][cd];
      buffer_2AA_grad[base_idx + 1] = root->data->Ln_ab[ab] + dw_p_ab[ab][cd];
      buffer_2AA_signs[base_idx + 1] = dw_p_ab_signs[ab][cd];
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(buffer_2AA_grad, buffer_2AA_signs, 2 * AA_ij);
    grad[A_i_p_A_j + cd] = logsumexp_result.sign * pow(2, logsumexp_result.result - fx);
  }
  deinitialize_node(root);
  return fx;
}

void initialize_constants(Constants* consts) {
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij_padded = pad_float(consts->AA_ij);
  consts->AA_ij_padded = AA_ij_padded;

  consts->p_ab = (c_float_t*) malloc(sizeof(c_float_t) * AA_ij);
  consts->dv_p_ab = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * AA_ij);
  consts->dv_p_ab_signs = (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * AA_ij);
  consts->dw_p_ab = (c_float_t*) calloc(AA_ij * AA_ij_padded, sizeof(c_float_t));
  consts->dw_p_ab_signs = (c_float_t*) malloc(sizeof(c_float_t) * AA_ij * AA_ij_padded);
  consts->p_ij_cond = (c_float_t*) calloc(AA_ij, sizeof(c_float_t));
  consts->dv_p_ij_cond = (c_float_t*) calloc(A_i_p_A_j * AA_ij, sizeof(c_float_t));
  consts->dv_p_ij_cond_signs =  (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * AA_ij);
  consts->dw_p_ij_cond = (c_float_t*) calloc(AA_ij * AA_ij_padded, sizeof(c_float_t));
  consts->dw_p_ij_cond_signs =  (c_float_t*) malloc(sizeof(c_float_t) * AA_ij * AA_ij_padded);
  consts->p_ji_cond = (c_float_t*) calloc(AA_ij, sizeof(c_float_t));
  consts->dv_p_ji_cond = (c_float_t*) calloc(A_i_p_A_j * AA_ij, sizeof(c_float_t));
  consts->dv_p_ji_cond_signs =  (c_float_t*) malloc(sizeof(c_float_t) * A_i_p_A_j * AA_ij);
  consts->dw_p_ji_cond = (c_float_t*) calloc(AA_ij * AA_ij_padded, sizeof(c_float_t));
  consts->dw_p_ji_cond_signs =  (c_float_t*) malloc(sizeof(c_float_t) * AA_ij * AA_ij_padded);
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

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij_padded = consts->AA_ij_padded;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij = consts->AA_ij;

  // p_ab related precomputations
  c_float_t total_sum = 0;
  c_float_t *p_ab = consts->p_ab;
  initialize_array(p_ab, log0, AA_ij);
  for (int a = 0; a < A_i; a++) {
    for (int b = 0; b < A_j; b++) {
      p_ab[a * A_j + b] = v[a] + v[A_i + b] + w[a * A_j + b];
      total_sum += p_ab[a * A_j + b];
    }
  }
  c_float_t normalization = logsumexpn(p_ab, AA_ij);
  for (int ab = 0; ab < AA_ij; ab++) {
    p_ab[ab] -= normalization;
  }

  c_float_t pi_a[A_i];
  for (int a = 0; a < A_i; a++) {
    pi_a[a] = logsumexpn(p_ab + a * A_j, A_j);
  }
  c_float_t pj_b[A_j];
  c_float_t a_tmp[A_i];
  for (int b = 0; b < A_j; b++) {
    for (int a = 0; a < A_i; a++) {
      a_tmp[a] = p_ab[a * A_j + b];
    }
    pj_b[b] = logsumexpn(a_tmp, A_i);
  }

  c_float_t *dv_p_ab = consts->dv_p_ab;
  initialize_array(dv_p_ab, log0, A_i_p_A_j * AA_ij);
  for (int c = 0; c < A_i; c++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        int ind = c * AA_ij + a * A_j + b;
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? c_f0: log0, c_f1,  pi_a[c], -c_f1);
        dv_p_ab[ind] = logsumexp_result.result + p_ab[a * A_j + b];
        consts->dv_p_ab_signs[ind] = logsumexp_result.sign;
      }
    }
  }
  for (int d = 0; d < A_j; d++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        int ind = (A_i + d) * AA_ij + a * A_j + b;
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? c_f0: log0, c_f1,  pj_b[d], -c_f1);
        dv_p_ab[ind] = logsumexp_result.result + p_ab[a * A_j + b];
        consts->dv_p_ab_signs[ind] = logsumexp_result.sign;
      }
    }
  }


  c_float_t *dw_p_ab_lin = consts->dw_p_ab;
  initialize_array(dw_p_ab_lin, log0, AA_ij * AA_ij_padded);
  c_float_t *dw_p_ab_signs_lin = consts->dw_p_ab_signs;
  c_float_t (*dw_p_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) dw_p_ab_lin;
  c_float_t (*dw_p_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) dw_p_ab_signs_lin;


  for (int c = 0; c < A_i; c++) {
    for (int d = 0; d < A_j; d++) {
      for (int a = 0; a < A_i; a++) {
        for (int b = 0; b < A_j; b++) {
          int cd = c*A_j + d;
          SignedLogExp logsumexp_result = signed_logsumexp2((a == c && b == d) ? c_f0: log0, c_f1, p_ab[c * A_j + d], -c_f1);
          dw_p_ab[a][b][cd] = logsumexp_result.result + p_ab[a * A_j + b];
          dw_p_ab_signs[a][b][cd] = logsumexp_result.sign;
        }
      }
    }
  }

  // p(a,.|.,b)
  c_float_t *p_ij_cond = consts->p_ij_cond;
  initialize_array(p_ij_cond, log0, AA_ij);
  c_float_t tmp_prob[(A_i > A_j) ? A_i : A_j];
  for (int b = 0; b < A_j; b++) {
    for (int a = 0; a < A_i; a++) {
      c_float_t log_prob = v[a] + w[a * A_j + b];
      tmp_prob[a] = log_prob;
      p_ij_cond[a * A_j + b] = log_prob;
    }
    c_float_t norm = logsumexpn(tmp_prob, A_i);
    for (int a = 0; a < A_i; a++) {
      p_ij_cond[a * A_j + b] -= norm;
    }
  }

  c_float_t *dv_p_ij_cond = consts->dv_p_ij_cond;
  initialize_array(dv_p_ij_cond, log0, A_i_p_A_j * AA_ij);
  for (int c = 0; c < A_i; c++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        int ind = c * AA_ij + a * A_j + b;
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? c_f0: log0, c_f1, p_ij_cond[c * A_j + b], -c_f1);
        dv_p_ij_cond[ind] = logsumexp_result.result + p_ij_cond[a * A_j + b];
        consts->dv_p_ij_cond_signs[ind] = logsumexp_result.sign;
      }
    }
  }

  initialize_array(consts->dw_p_ij_cond, log0, AA_ij * AA_ij_padded);
  c_float_t (*dw_p_ij_cond)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ij_cond;
  c_float_t (*dw_p_ij_cond_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ij_cond_signs;

  for (int c = 0; c < A_i; c++) {
    for (int d = 0; d < A_j; d++) {
      for (int a = 0; a < A_i; a++) {
        int b = d;
        int cd = c*A_j + d;
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? c_f0: log0, c_f1, p_ij_cond[c * A_j + d], -c_f1);
        dw_p_ij_cond[a][b][cd] = logsumexp_result.result + p_ij_cond[a * A_j + b];
        dw_p_ij_cond_signs[a][b][cd] = logsumexp_result.sign;
      }
    }
  }

  // p(.,b|a,.)
  c_float_t *p_ji_cond = consts->p_ji_cond;
  initialize_array(p_ji_cond, log0, AA_ij);
  for (int a = 0; a < A_i; a++) {
    for (int b = 0; b < A_j; b++) {
      c_float_t prob = v[A_i + b] + w[a * A_j + b];
      p_ji_cond[a * A_j + b] = prob;
      tmp_prob[b] = prob;
    }
    c_float_t normalization = logsumexpn(tmp_prob, A_j);
    for (int b = 0; b < A_j; b++) {
      p_ji_cond[a * A_j + b] -= normalization;
    }
  }

  c_float_t *dv_p_ji_cond = consts->dv_p_ji_cond;
  initialize_array(consts->dv_p_ji_cond, log0, A_i_p_A_j * AA_ij);
  for (int d = 0; d < A_j; d++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        int ind = A_i * AA_ij + d * AA_ij + b * A_i + a;
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? c_f0: log0, c_f1, p_ji_cond[a * A_j + d], -c_f1);
        dv_p_ji_cond[ind] = logsumexp_result.result + p_ji_cond[a * A_j + b];
        consts->dv_p_ji_cond_signs[ind] = logsumexp_result.sign;
      }
    }
  }

  c_float_t (*dw_p_ji_cond)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ji_cond;
  c_float_t (*dw_p_ji_cond_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ji_cond_signs;

  initialize_array(consts->dw_p_ji_cond, log0, AA_ij * AA_ij_padded);
  for (int c = 0; c < A_i; c++) {
    for (int d = 0; d < A_j; d++) {
      for (int b = 0; b < A_j; b++) {
        int a = c;
        int cd = c*A_j + d;
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? c_f0: log0, c_f1, p_ji_cond[c * A_j + d], -c_f1);
        dw_p_ji_cond[a][b][cd] = logsumexp_result.result + p_ji_cond[a * A_j + b];
        dw_p_ji_cond_signs[a][b][cd] = logsumexp_result.sign;
      }
    }
  }
}
