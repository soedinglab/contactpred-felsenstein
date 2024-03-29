#include "felsenstein_simd.h"
#include <stdlib.h>
#include <math.h>

#ifdef SINGLE_PRECISION
#include "simd_functions_ps.h"
#else
#include "simd_functions_pd.h"
#endif

#ifdef DEBUG_PRINT
#include "debug_tools.h"
#endif

void initialize_leaf(Node* leaf, Constants* consts) {

  int A_j = consts->A_j;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;
  int AA_ij_padded = consts->AA_ij_padded;

  uint8_t* msa = consts->msa;
  int L = consts->L;
  int i = consts->i;
  int j = consts->j;
  int a = msa[leaf->seq_id * L + i];
  int b = msa[leaf->seq_id * L + j];

  initialize_node(leaf, consts);
  initialize_array(leaf->data->Ln_ab, log0, AA_ij);
  leaf->data->Ln_ab[a * A_j + b] = 0;

  initialize_array(leaf->data->dv_Ln_ab, log0, A_i_p_A_j_padded * AA_ij);
  initialize_array(leaf->data->dw_Ln_ab, log0, AA_ij * AA_ij_padded);
  initialize_array(leaf->data->dv_Ln_ab_signs, c_f1, A_i_p_A_j_padded * AA_ij);
  initialize_array(leaf->data->dw_Ln_ab_signs, c_f1, AA_ij * AA_ij_padded);
}

void initialize_nodebuffer(NodeBuffer* buffer, Constants* consts) {

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;
  int AA_ij_padded = consts->AA_ij_padded;

  buffer->Ln_ia = (c_float_t*) malloc(sizeof(c_float_t) * A_i);
  buffer->Ln_jb = (c_float_t*) malloc(sizeof(c_float_t) * A_j);

  buffer->dv_Ln = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded);
  buffer->dv_Ln_signs = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded);
  buffer->dv_Ln_ia = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * A_i);
  buffer->dv_Ln_ia_signs = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * A_i);
  buffer->dv_Ln_jb = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * A_j);
  buffer->dv_Ln_jb_signs = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * A_j);

  buffer->dw_Ln = malloc_simd_farr(sizeof(c_float_t)*AA_ij_padded);
  buffer->dw_Ln_signs = malloc_simd_farr(sizeof(c_float_t)*AA_ij_padded);
  buffer->dw_Ln_ia = malloc_simd_farr(sizeof(c_float_t) * A_i * AA_ij_padded);
  buffer->dw_Ln_ia_signs = malloc_simd_farr(sizeof(c_float_t) * A_i * AA_ij_padded);
  buffer->dw_Ln_jb = malloc_simd_farr(sizeof(c_float_t) * A_j * AA_ij_padded);
  buffer->dw_Ln_jb_signs = malloc_simd_farr(sizeof(c_float_t) * A_j * AA_ij_padded);

  buffer->dw_mut0_buffer = malloc_simd_farr(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_mut1_buffer = malloc_simd_farr(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_mut1_sign_buffer = malloc_simd_farr(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_mut2_buffer = malloc_simd_farr(AA_ij_padded*sizeof(c_float_t));

  buffer->dv_mut0_buffer = malloc_simd_farr(A_i_p_A_j_padded*sizeof(c_float_t));
  buffer->dv_mut1_buffer = malloc_simd_farr(A_i_p_A_j_padded*sizeof(c_float_t));
  buffer->dv_mut1_sign_buffer = malloc_simd_farr(A_i_p_A_j_padded*sizeof(c_float_t));
  buffer->dv_mut2_buffer = malloc_simd_farr(A_i_p_A_j_padded*sizeof(c_float_t));

}

void initialize_buffer(Buffer* buffer, Constants* consts) {
  int AA_ij_padded = consts->AA_ij_padded;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;
  buffer->left = malloc(sizeof(NodeBuffer));
  initialize_nodebuffer(buffer->left, consts);
  buffer->right = malloc(sizeof(NodeBuffer));
  initialize_nodebuffer(buffer->right, consts);

  buffer->dw_left_Lab = malloc_simd_farr(AA_ij_padded*sizeof(c_float_t));
  buffer->dw_right_Lab = malloc_simd_farr(AA_ij_padded*sizeof(c_float_t));

  buffer->dv_left_Lab = malloc_simd_farr(A_i_p_A_j_padded*sizeof(c_float_t));
  buffer->dv_right_Lab = malloc_simd_farr(A_i_p_A_j_padded*sizeof(c_float_t));

  buffer->dv_logexp_buffer = malloc(sizeof(LogExpBuffer));
  initialize_logexpbuffer(buffer->dv_logexp_buffer, consts, A_i_p_A_j_padded);
  buffer->dw_logexp_buffer = malloc(sizeof(LogExpBuffer));
  initialize_logexpbuffer(buffer->dw_logexp_buffer, consts, AA_ij_padded);

}

void deinitialize_buffer(Buffer* buffer) {
  deinitialize_nodebuffer(buffer->left);
  deinitialize_nodebuffer(buffer->right);
  free(buffer->left);
  free(buffer->right);

  free(buffer->dw_left_Lab);
  free(buffer->dw_right_Lab);
  free(buffer->dv_left_Lab);
  free(buffer->dv_right_Lab);

  deinitialize_logexpbuffer(buffer->dv_logexp_buffer);
  free(buffer->dv_logexp_buffer);
  deinitialize_logexpbuffer(buffer->dw_logexp_buffer);
  free(buffer->dw_logexp_buffer);
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

  free(buffer->dw_mut0_buffer);
  free(buffer->dw_mut1_buffer);
  free(buffer->dw_mut1_sign_buffer);
  free(buffer->dw_mut2_buffer);

  free(buffer->dv_mut0_buffer);
  free(buffer->dv_mut1_buffer);
  free(buffer->dv_mut1_sign_buffer);
  free(buffer->dv_mut2_buffer);
}

void precompute_buffer(NodeBuffer* node_buffer, NodePrecomputation* data, Constants* consts, Buffer* buffer){

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;
  int AA_ij_padded = consts->AA_ij_padded;

  c_float_t (* dv_Ln_ab)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) data->dv_Ln_ab;
  c_float_t (* dv_Ln_ab_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) data->dv_Ln_ab_signs;

  c_float_t (*dv_p_ab)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ab;
  c_float_t (*dv_p_ab_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ab_signs;
  c_float_t (*p_ij_cond)[A_j] = (c_float_t (*)[A_j]) consts->p_ij_cond;

  c_float_t (*dv_p_ij_cond)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ij_cond;
  c_float_t (*dv_p_ij_cond_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ij_cond_signs;

  c_float_t (*p_ji_cond)[A_j] =  (c_float_t (*)[A_j]) consts->p_ji_cond;
  c_float_t (*dv_p_ji_cond)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ji_cond;
  c_float_t (*dv_p_ji_cond_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ji_cond_signs;
  c_float_t (*dv_Ln_ia)[A_i_p_A_j_padded] =  (c_float_t (*)[A_i_p_A_j_padded]) node_buffer->dv_Ln_ia;
  c_float_t (*dv_Ln_ia_signs)[A_i_p_A_j_padded] =  (c_float_t (*)[A_i_p_A_j_padded]) node_buffer->dv_Ln_ia_signs;
  c_float_t (*dv_Ln_jb)[A_i_p_A_j_padded] =  (c_float_t (*)[A_i_p_A_j_padded]) node_buffer->dv_Ln_jb;
  c_float_t (*dv_Ln_jb_signs)[A_i_p_A_j_padded] =  (c_float_t (*)[A_i_p_A_j_padded]) node_buffer->dv_Ln_jb_signs;


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

  c_float_t (*dw_Ln_ia)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) node_buffer->dw_Ln_ia;
  c_float_t (*dw_Ln_ia_signs)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) node_buffer->dw_Ln_ia_signs;
  c_float_t (*dw_Ln_jb)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) node_buffer->dw_Ln_jb;
  c_float_t (*dw_Ln_jb_signs)[AA_ij_padded] =  (c_float_t (*)[AA_ij_padded]) node_buffer->dw_Ln_jb_signs;


  // Nulling out buffer
  node_buffer->Ln = 0;

  c_float_t log_buffer_AA[AA_ij];
  c_float_t log_buffer_A_j[A_j];
  c_float_t log_buffer_A_i[A_i];


  #ifdef DEBUG_NOSIMD
  int base_idx;
  c_float_t log_buffer_2A_i[2 * A_i];
  c_float_t sign_buffer_2A_i[2 * A_i];
  c_float_t log_buffer_2A_j[2 * A_j];
  c_float_t sign_buffer_2A_j[2 * A_j];
  c_float_t log_buffer_2AA[2 * AA_ij];
  c_float_t sign_buffer_2AA[2 * AA_ij];
  #endif

  // d/dp p(Xm)
  for(int a = 0; a < A_i; a++) {
    for(int b = 0; b < A_j; b++) {
      log_buffer_AA[a*A_j + b] = data->Ln_ab[a*A_j + b] + p_ab[a][b];
    }
  }
  node_buffer->Ln = logsumexpn(log_buffer_AA, AA_ij);

  #ifndef DEBUG_NOSIMD
  logsumexp_matrix_ax01(A_i, A_j, A_i_p_A_j_padded,
                        node_buffer->dv_Ln, node_buffer->dv_Ln_signs,
                        dv_Ln_ab, dv_Ln_ab_signs, p_ab,
                        dv_p_ab, dv_p_ab_signs, Ln_ab,
                        buffer->dv_logexp_buffer);

  logsumexp_matrix_ax01(A_i, A_j, AA_ij_padded,
                        node_buffer->dw_Ln, node_buffer->dw_Ln_signs,
                        dw_Ln_ab, dw_Ln_ab_signs, p_ab,
                        dw_p_ab, dw_p_ab_signs, Ln_ab,
                        buffer->dw_logexp_buffer);
  #else
  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for(int c_p = 0; c_p < A_i; c_p++) {
      for(int d_p = 0; d_p < A_j; d_p++) {
        int base_idx = 2*(c_p*A_j + d_p);
        log_buffer_2AA[base_idx] = dv_Ln_ab[c_p][d_p][lc] + p_ab[c_p][d_p];
        sign_buffer_2AA[base_idx] = dv_Ln_ab_signs[c_p][d_p][lc];
        log_buffer_2AA[base_idx + 1] = data->Ln_ab[c_p*A_j + d_p] + dv_p_ab[c_p][d_p][lc];
        sign_buffer_2AA[base_idx + 1] = dv_p_ab_signs[c_p][d_p][lc];
      }
    }
    SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2AA, sign_buffer_2AA, 2 * AA_ij);
    node_buffer->dv_Ln[lc] = logsumexp_result.result;
    node_buffer->dv_Ln_signs[lc] = logsumexp_result.sign;
  }

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
    node_buffer->dw_Ln[cd] = logsumexp_result.result;
    node_buffer->dw_Ln_signs[cd] = logsumexp_result.sign;
  }
  #endif


  // p(Xm|a, .)
  for(int a = 0; a < A_i; a++) {
    for(int d = 0; d < A_j; d++) {
      log_buffer_A_j[d] = data->Ln_ab[a * A_j + d] + p_ji_cond[a][d];
    }
    node_buffer->Ln_ia[a] = logsumexpn(log_buffer_A_j, A_j);
  }

  #ifndef DEBUG_NOSIMD
  logsumexp_matrix_ax1(A_i, A_j, A_i_p_A_j_padded,
    dv_Ln_ia, dv_Ln_ia_signs,
    dv_Ln_ab, dv_Ln_ab_signs, p_ji_cond,
    dv_p_ji_cond, dv_p_ji_cond_signs, Ln_ab,
    buffer->dv_logexp_buffer
  );

  logsumexp_matrix_ax1(A_i, A_j, AA_ij_padded,
    dw_Ln_ia, dw_Ln_ia_signs,
    dw_Ln_ab, dw_Ln_ab_signs, p_ji_cond,
    dw_p_ji_cond, dw_p_ji_cond_signs, Ln_ab,
    buffer->dw_logexp_buffer
  );
  #else
  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for (int a = 0; a < A_i; a++) {
      base_idx = 0;
      for (int d_p = 0; d_p < A_j; d_p++) {
        log_buffer_2A_j[base_idx] = dv_Ln_ab[a][d_p][lc] + p_ji_cond[a][d_p];
        sign_buffer_2A_j[base_idx] = dv_Ln_ab_signs[a][d_p][lc];
        base_idx++;
        if(lc >= A_i) {
          log_buffer_2A_j[base_idx] = data->Ln_ab[a * A_j + d_p] + dv_p_ji_cond[a][d_p][lc];
          sign_buffer_2A_j[base_idx] = dv_p_ji_cond_signs[a][d_p][lc];
          base_idx++;
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_j, sign_buffer_2A_j, base_idx);
      dv_Ln_ia[a][lc] = logsumexp_result.result;
      dv_Ln_ia_signs[a][lc] = logsumexp_result.sign;
    }
  }

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
  #endif


  // d/dp p(Xm|.,b)
  for(int b = 0; b < A_j; b++) {
    for(int c = 0; c < A_i; c++) {
      log_buffer_A_i[c] = data->Ln_ab[c * A_j + b] + p_ij_cond[c][b];
    }
    node_buffer->Ln_jb[b] = logsumexpn(log_buffer_A_i, A_i);
  }

  #ifndef DEBUG_NOSIMD
  logsumexp_matrix_ax0(A_i, A_j, A_i_p_A_j_padded,
                       dv_Ln_jb, dv_Ln_jb_signs,
                       dv_Ln_ab, dv_Ln_ab_signs, p_ij_cond,
                       dv_p_ij_cond, dv_p_ij_cond_signs, Ln_ab,
                       buffer->dv_logexp_buffer
  );

  logsumexp_matrix_ax0(A_i, A_j, AA_ij_padded,
                       dw_Ln_jb, dw_Ln_jb_signs,
                       dw_Ln_ab, dw_Ln_ab_signs, p_ij_cond,
                       dw_p_ij_cond, dw_p_ij_cond_signs, Ln_ab,
                       buffer->dw_logexp_buffer
  );
  #else
  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for (int b = 0; b < A_j; b++) {
      base_idx = 0;
      for (int c_p = 0; c_p < A_i; c_p++) {
        log_buffer_2A_i[base_idx] = dv_Ln_ab[c_p][b][lc] + p_ij_cond[c_p][b];
        sign_buffer_2A_i[base_idx] = dv_Ln_ab_signs[c_p][b][lc];
        base_idx++;
        if(lc < A_i) {
          log_buffer_2A_i[base_idx] = data->Ln_ab[c_p * A_j + b] + dv_p_ij_cond[c_p][b][lc];
          sign_buffer_2A_i[base_idx] = dv_p_ij_cond_signs[c_p][b][lc];
          base_idx++;
        }
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_i, sign_buffer_2A_i, base_idx);
      dv_Ln_jb[b][lc] = logsumexp_result.result;
      dv_Ln_jb_signs[b][lc] = logsumexp_result.sign;
    }
  }

  for(int cd = 0; cd < AA_ij; cd++) {
    for (int b = 0; b < A_j; b++) {
      for (int c_p = 0; c_p < A_i; c_p++) {
        int base_idx = 2*c_p;
        log_buffer_2A_i[base_idx] = dw_Ln_ab[c_p][b][cd] + p_ij_cond[c_p][b];
        sign_buffer_2A_i[base_idx] = dw_Ln_ab_signs[c_p][b][cd];
        log_buffer_2A_i[base_idx + 1] = data->Ln_ab[c_p*A_j + b] + dw_p_ij_cond[c_p][b][cd];
        sign_buffer_2A_i[base_idx + 1] = dw_p_ij_cond_signs[c_p][b][cd];
      }
      SignedLogExp logsumexp_result = signed_logsumexp_n(log_buffer_2A_i, sign_buffer_2A_i, 2*A_i);
      dw_Ln_jb[b][cd] = logsumexp_result.result;
      dw_Ln_jb_signs[b][cd] = logsumexp_result.sign;
    }
  }
  #endif
}

void initialize_node(Node* node, Constants* consts) {

  int AA_ij = consts->AA_ij;
  int AA_ij_padded = consts->AA_ij_padded;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;

  NodePrecomputation* data = malloc(sizeof(NodePrecomputation));
  data->Ln_ab = malloc_simd_farr(sizeof(c_float_t) * AA_ij);
  data->dv_Ln_ab = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * AA_ij);
  data->dv_Ln_ab_signs = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * AA_ij);
  data->dw_Ln_ab = malloc_simd_farr(sizeof(c_float_t) * AA_ij * AA_ij_padded);
  data->dw_Ln_ab_signs = malloc_simd_farr(sizeof(c_float_t) * AA_ij * AA_ij_padded);
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

  int A_j = consts->A_j;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;
  int AA_ij_padded = consts->AA_ij_padded;

  #ifdef DEBUG_NOSIMD
  int A_i_p_A_j = consts->A_i_p_A_j;
  int AA_ij = consts->AA_ij;
  #endif

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


  c_float_t (*dv_Ln_ia)[A_i_p_A_j_padded] = (c_float_t (*)[A_i_p_A_j_padded]) child_buffer->dv_Ln_ia;
  c_float_t (*dv_Ln_ia_signs)[A_i_p_A_j_padded] = (c_float_t (*)[A_i_p_A_j_padded]) child_buffer->dv_Ln_ia_signs;
  c_float_t (*dv_Ln_jb)[A_i_p_A_j_padded] = (c_float_t (*)[A_i_p_A_j_padded]) child_buffer->dv_Ln_jb;
  c_float_t (*dv_Ln_jb_signs)[A_i_p_A_j_padded] = (c_float_t (*)[A_i_p_A_j_padded]) child_buffer->dv_Ln_jb_signs;
  c_float_t (*dv_Ln_ab)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) child_data->dv_Ln_ab;
  c_float_t (*dv_Ln_ab_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) child_data->dv_Ln_ab_signs;

  #ifndef DEBUG_NOSIMD
  add_constant(buffer->dv_mut2_buffer,  child_buffer->dv_Ln, 2*log_1mr, A_i_p_A_j_padded);
  signedlogsumexp2_array(buffer->dv_mut1_buffer, buffer->dv_mut1_sign_buffer,
                         dv_Ln_ia[a], dv_Ln_ia_signs[a],
                         dv_Ln_jb[b],  dv_Ln_jb_signs[b],
                         A_i_p_A_j_padded
  );
  add_constant(buffer->dv_mut1_buffer, buffer->dv_mut1_buffer, log_r + log_1mr,  A_i_p_A_j_padded);
  add_constant(buffer->dv_mut0_buffer, dv_Ln_ab[a][b], 2*log_r, A_i_p_A_j_padded);
  signedlogsumexp3_array(dv_L_ab, dv_L_ab_signs,
                         buffer->dv_mut2_buffer, child_buffer->dv_Ln_signs,
                         buffer->dv_mut1_buffer, buffer->dv_mut1_sign_buffer,
                         buffer->dv_mut0_buffer, dv_Ln_ab_signs[a][b],
                         A_i_p_A_j_padded
  );
  #else
  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    c_float_t ddv_mut2 = 2*log_1mr + child_buffer->dv_Ln[lc];
    c_float_t ddv_mut2_sign = child_buffer->dv_Ln_signs[lc];

    c_float_t dv_Ln_ia_val = dv_Ln_ia[a][lc];
    c_float_t dv_Ln_ia_sign_val = dv_Ln_ia_signs[a][lc];
    c_float_t dv_Ln_jb_val = dv_Ln_jb[b][lc];
    c_float_t dv_Ln_jb_sign_val = dv_Ln_jb_signs[b][lc];;
    SignedLogExp mut1_logsumexp = signed_logsumexp2(dv_Ln_ia_val, dv_Ln_ia_sign_val, dv_Ln_jb_val, dv_Ln_jb_sign_val);
    c_float_t ddv_mut1 = log_r + log_1mr + mut1_logsumexp.result;
    c_float_t ddv_mut1_sign = mut1_logsumexp.sign;

    c_float_t ddv_mut0 = 2*log_r + dv_Ln_ab[a][b][lc];
    c_float_t ddv_mut0_sign = dv_Ln_ab_signs[a][b][lc];

    SignedLogExp mut_logsumexp = signed_logsumexp3(ddv_mut2, ddv_mut2_sign, ddv_mut1, ddv_mut1_sign, ddv_mut0, ddv_mut0_sign);
    dv_L_ab[lc] = mut_logsumexp.result;
    dv_L_ab_signs[lc] = mut_logsumexp.sign;
  }
  #endif


  c_float_t (*dw_Ln_ia)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_ia;
  c_float_t (*dw_Ln_ia_signs)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_ia_signs;
  c_float_t (*dw_Ln_jb)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_jb;
  c_float_t (*dw_Ln_jb_signs)[AA_ij_padded] = (c_float_t (*)[AA_ij_padded]) child_buffer->dw_Ln_jb_signs;
  c_float_t (*dw_Ln_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) child_data->dw_Ln_ab;
  c_float_t (*dw_Ln_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) child_data->dw_Ln_ab_signs;

  #ifndef DEBUG_NOSIMD
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
  #else
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
  #endif

}

void recurse_tree(Node* node, Constants* consts, Buffer* buffer) {

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij_padded = consts->AA_ij_padded;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;

  if (node->left == NULL && node->right == NULL) {
    // this is a leaf node
    initialize_leaf(node, consts);
    return;
  } else {
    recurse_tree(node->left, consts, buffer);
    recurse_tree(node->right, consts, buffer);
  }
  initialize_node(node, consts);

  c_float_t dv_left_Lab[A_i_p_A_j_padded];
  c_float_t dv_left_Lab_signs[A_i_p_A_j_padded];
  c_float_t dv_right_Lab[A_i_p_A_j_padded];
  c_float_t dv_right_Lab_signs[A_i_p_A_j_padded];

  c_float_t dw_left_Lab[AA_ij_padded];
  c_float_t dw_left_Lab_signs[AA_ij_padded];
  c_float_t dw_right_Lab[AA_ij_padded];
  c_float_t dw_right_Lab_signs[AA_ij_padded];


  // precalculate aggregated values
  if(node->left != NULL) {
    NodePrecomputation *left_data = node->left->data;
    NodeBuffer* left_buf = buffer->left;
    precompute_buffer(left_buf, left_data, consts, buffer);
  }
  if(node->right != NULL) {
    NodePrecomputation* right_data = node->right->data;
    NodeBuffer* right_buf = buffer->right;
    precompute_buffer(right_buf, right_data, consts, buffer);
  }

  NodePrecomputation *node_data = node->data;
  c_float_t (*dv_Ln_ab)[A_j][A_i_p_A_j_padded] = (c_float_t(*)[A_j][A_i_p_A_j_padded]) node_data->dv_Ln_ab;
  c_float_t (*dv_Ln_ab_signs)[A_j][A_i_p_A_j_padded] = (c_float_t(*)[A_j][A_i_p_A_j_padded]) node_data->dv_Ln_ab_signs;

  c_float_t (*dw_Ln_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) node_data->dw_Ln_ab;
  c_float_t (*dw_Ln_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) node_data->dw_Ln_ab_signs;

  for (int a = 0; a < A_i; a++) {
    for (int b = 0; b < A_j; b++) {
      c_float_t left_Lab = 0;
      c_float_t right_Lab = 0;

      if (node->left != NULL) {
        initialize_array(dv_left_Lab, c_f0, A_i_p_A_j_padded);
        initialize_array(dw_left_Lab, c_f0, AA_ij_padded);
        compute_Ln_branch(node->left, node->log_p_left, node->log_1mp_left, buffer->left, consts, &left_Lab, dv_left_Lab, dv_left_Lab_signs,
                          dw_left_Lab, dw_left_Lab_signs, a, b);
      }
      if (node->right != NULL) {
        initialize_array(dv_right_Lab, c_f0, A_i_p_A_j_padded);
        initialize_array(dw_right_Lab, c_f0, AA_ij_padded);
        compute_Ln_branch(node->right, node->log_p_right, node->log_1mp_right, buffer->right, consts, &right_Lab, dv_right_Lab,
                          dv_right_Lab_signs, dw_right_Lab, dw_right_Lab_signs, a, b);
      }

      c_float_t Ln_ab = left_Lab + right_Lab;

      node_data->Ln_ab[a * A_j + b] = Ln_ab;

      // combine derivatives for left and right children

      add_constant(buffer->dv_left_Lab, dv_left_Lab, right_Lab, A_i_p_A_j_padded);
      add_constant(buffer->dv_right_Lab, dv_right_Lab, left_Lab, A_i_p_A_j_padded);
      signedlogsumexp2_array(dv_Ln_ab[a][b], dv_Ln_ab_signs[a][b],
                             buffer->dv_left_Lab, dv_left_Lab_signs,
                             buffer->dv_right_Lab, dv_right_Lab_signs,
                             A_i_p_A_j_padded
      );

      add_constant(buffer->dw_left_Lab, dw_left_Lab, right_Lab, AA_ij_padded);
      add_constant(buffer->dw_right_Lab, dw_right_Lab, left_Lab, AA_ij_padded);
      signedlogsumexp2_array(dw_Ln_ab[a][b], dw_Ln_ab_signs[a][b],
                             buffer->dw_left_Lab, dw_left_Lab_signs,
                             buffer->dw_right_Lab, dw_right_Lab_signs,
                             AA_ij_padded
        );
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

  c_float_t loge_2 = log(2);
  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij = consts->AA_ij;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;
  int AA_ij_padded = consts->AA_ij_padded;

  precalculate_constants(consts, x, x + A_i_p_A_j);
  Node* root = consts->phylo_tree;

  recurse_tree(root, consts, buf);
  c_float_t buffer_AA_fx[AA_ij];

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("root p_ab", consts->p_ab, AA_ij);
  print_array_dbg_loc("root Ln_ab", root->data->Ln_ab, AA_ij);
  #endif

  c_float_t fx = 0;
  for(int a = 0; a < A_i; a++) {
    for(int b = 0; b < A_j; b++) {
      buffer_AA_fx[a * A_j + b] = root->data->Ln_ab[a * A_j + b] + consts->p_ab[a * A_j + b];
    }
  }
  fx = logsumexpn(buffer_AA_fx, AA_ij);

  #ifdef DEBUG_PRINT
  printf("DBG: (fx) %g\n", fx);
  #endif

  initialize_array(grad, c_f0, A_i_p_A_j + AA_ij);
  c_float_t buffer_2AA_grad[2 * AA_ij];
  c_float_t buffer_2AA_signs[2 * AA_ij];

  c_float_t (*dv_Ln_ab)[A_i_p_A_j_padded] = (c_float_t(*)[A_i_p_A_j_padded]) root->data->dv_Ln_ab;
  c_float_t (*dv_Ln_ab_signs)[A_i_p_A_j_padded] = (c_float_t(*)[A_i_p_A_j_padded]) root->data->dv_Ln_ab_signs;
  c_float_t (*dv_p_ab)[A_i_p_A_j_padded] = (c_float_t(*)[A_i_p_A_j_padded]) consts->dv_p_ab;
  c_float_t (*dv_p_ab_signs)[A_i_p_A_j_padded] = (c_float_t(*)[A_i_p_A_j_padded]) consts->dv_p_ab_signs;

  for(int lc = 0; lc < A_i_p_A_j; lc++) {
    for(int ab = 0; ab < AA_ij; ab++) {
      int base_idx = 2*ab;
      buffer_2AA_grad[base_idx] = dv_Ln_ab[ab][lc] + consts->p_ab[ab];
      buffer_2AA_signs[base_idx] = dv_Ln_ab_signs[ab][lc];
      buffer_2AA_grad[base_idx + 1] = root->data->Ln_ab[ab] + dv_p_ab[ab][lc];
      buffer_2AA_signs[base_idx + 1] = dv_p_ab_signs[ab][lc];
    }
    #ifdef DEBUG_PRINT
    print_array_dbg_loc("buffer_2AA_grad", buffer_2AA_grad, AA_ij);
    print_array_dbg_loc("buffer_2AA_signs", buffer_2AA_signs, AA_ij);
    #endif

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

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("grad", grad, AA_ij + A_i_p_A_j);
  #endif

  deinitialize_node(root);
  return fx * loge_2;
}

void initialize_constants(Constants* consts) {

  int AA_ij = consts->A_i * consts->A_j;
  consts->AA_ij = AA_ij;
  int A_i_p_A_j = consts->A_i + consts->A_j;
  consts->A_i_p_A_j = A_i_p_A_j;

  int AA_ij_padded = simd_padded(consts->AA_ij);
  consts->AA_ij_padded = AA_ij_padded;

  int A_i_p_A_j_padded = simd_padded(consts->A_i_p_A_j);
  consts->A_i_p_A_j_padded = A_i_p_A_j_padded;

  #ifdef DEBUG_PRINT
  printf("DBG: (AA_ij) %d\n", consts->AA_ij);
  printf("DBG: (A_i_p_A_j) %d\n", consts->A_i_p_A_j);
  printf("DBG: (AA_ij_padded) %d\n", consts->AA_ij_padded);
  printf("DBG: (A_i_p_A_j_padded) %d\n", consts->A_i_p_A_j_padded);
  #endif

  consts->p_ab = (c_float_t*) malloc(sizeof(c_float_t) * AA_ij);
  consts->dv_p_ab = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * AA_ij);
  consts->dv_p_ab_signs = malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * AA_ij);
  consts->dw_p_ab = malloc_simd_farr(AA_ij * AA_ij_padded * sizeof(c_float_t));
  initialize_array(consts->dw_p_ab, c_f0, AA_ij * AA_ij_padded);
  consts->dw_p_ab_signs = malloc_simd_farr(sizeof(c_float_t) * AA_ij * AA_ij_padded);
  consts->p_ij_cond = (c_float_t*) calloc(AA_ij, sizeof(c_float_t));
  consts->dv_p_ij_cond = malloc_simd_farr(A_i_p_A_j_padded * AA_ij * sizeof(c_float_t));
  initialize_array(consts->dv_p_ij_cond, c_f0, A_i_p_A_j_padded * AA_ij);
  consts->dv_p_ij_cond_signs =  malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * AA_ij);
  consts->dw_p_ij_cond = malloc_simd_farr(AA_ij * AA_ij_padded * sizeof(c_float_t));
  initialize_array(consts->dw_p_ij_cond, c_f0, AA_ij * AA_ij_padded);
  consts->dw_p_ij_cond_signs =  malloc_simd_farr(sizeof(c_float_t) * AA_ij * AA_ij_padded);
  consts->p_ji_cond = (c_float_t*) calloc(AA_ij, sizeof(c_float_t));
  consts->dv_p_ji_cond = malloc_simd_farr(A_i_p_A_j_padded * AA_ij * sizeof(c_float_t));
  initialize_array(consts->dv_p_ji_cond, c_f0, A_i_p_A_j_padded * AA_ij);
  consts->dv_p_ji_cond_signs =  malloc_simd_farr(sizeof(c_float_t) * A_i_p_A_j_padded * AA_ij);
  consts->dw_p_ji_cond = malloc_simd_farr(AA_ij * AA_ij_padded * sizeof(c_float_t));
  initialize_array(consts->dw_p_ji_cond, c_f0, AA_ij * AA_ij_padded);
  consts->dw_p_ji_cond_signs =  malloc_simd_farr(sizeof(c_float_t) * AA_ij * AA_ij_padded);
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

  c_float_t loge_2 = log(2);

  int A_i = consts->A_i;
  int A_j = consts->A_j;
  int AA_ij_padded = consts->AA_ij_padded;
  int A_i_p_A_j = consts->A_i_p_A_j;
  int A_i_p_A_j_padded = consts->A_i_p_A_j_padded;
  int AA_ij = consts->AA_ij;

  // p_ab related precomputations
  c_float_t total_sum = 0;
  c_float_t *p_ab = consts->p_ab;
  initialize_array(p_ab, log0, AA_ij);
  for (int a = 0; a < A_i; a++) {
    for (int b = 0; b < A_j; b++) {
      p_ab[a * A_j + b] = (v[a] + v[A_i + b] + w[a * A_j + b]) / loge_2;
      total_sum += p_ab[a * A_j + b];
    }
  }
  c_float_t normalization = logsumexpn(p_ab, AA_ij);
  for (int ab = 0; ab < AA_ij; ab++) {
    p_ab[ab] -= normalization;
  }

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("p_ab", p_ab, AA_ij);
  #endif

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

  initialize_array(consts->dv_p_ab, log0, A_i_p_A_j_padded * AA_ij);
  initialize_array(consts->dv_p_ab_signs, c_f1, A_i_p_A_j_padded * AA_ij);
  c_float_t (*dv_p_ab)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ab;
  c_float_t (*dv_p_ab_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ab_signs;

  for (int c = 0; c < A_i; c++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? c_f0: log0, c_f1,  pi_a[c], -c_f1);
        dv_p_ab[a][b][c] = logsumexp_result.result + p_ab[a * A_j + b];
        dv_p_ab_signs[a][b][c] = logsumexp_result.sign;
      }
    }
  }
  for (int d = 0; d < A_j; d++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? c_f0: log0, c_f1,  pj_b[d], -c_f1);
        dv_p_ab[a][b][A_i + d] = logsumexp_result.result + p_ab[a * A_j + b];
        dv_p_ab_signs[a][b][A_i + d] = logsumexp_result.sign;
      }
    }
  }

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("dv_p_ab", dv_p_ab, A_i_p_A_j*AA_ij);
  print_array_dbg_loc("dv_p_ab_signs", consts->dv_p_ab_signs, A_i_p_A_j*AA_ij);
  #endif


  initialize_array(consts->dw_p_ab, log0, AA_ij * AA_ij_padded);
  initialize_array(consts->dw_p_ab_signs, c_f1, AA_ij * AA_ij_padded);
  c_float_t (*dw_p_ab)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ab;
  c_float_t (*dw_p_ab_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ab_signs;


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

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("dw_p_ab", dw_p_ab, AA_ij*AA_ij);
  print_array_dbg_loc("dw_p_ab_signs", consts->dw_p_ab_signs, AA_ij*AA_ij);
  #endif

  // p(a,.|.,b)
  c_float_t *p_ij_cond = consts->p_ij_cond;
  initialize_array(p_ij_cond, log0, AA_ij);
  c_float_t tmp_prob[(A_i > A_j) ? A_i : A_j];
  for (int b = 0; b < A_j; b++) {
    for (int a = 0; a < A_i; a++) {
      c_float_t log_prob = (v[a] + w[a * A_j + b])  / loge_2;
      tmp_prob[a] = log_prob;
      p_ij_cond[a * A_j + b] = log_prob;
    }
    c_float_t norm = logsumexpn(tmp_prob, A_i);
    for (int a = 0; a < A_i; a++) {
      p_ij_cond[a * A_j + b] -= norm;
    }
  }

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("p_ij_cond", p_ij_cond, AA_ij);
  #endif


  initialize_array(consts->dv_p_ij_cond, log0, AA_ij * A_i_p_A_j_padded);
  initialize_array(consts->dv_p_ij_cond_signs, c_f1, AA_ij * A_i_p_A_j_padded);
  c_float_t (*dv_p_ij_cond)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ij_cond;
  c_float_t (*dv_p_ij_cond_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ij_cond_signs;

  for (int c = 0; c < A_i; c++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        SignedLogExp logsumexp_result = signed_logsumexp2((a == c) ? c_f0: log0, c_f1, p_ij_cond[c * A_j + b], -c_f1);
        dv_p_ij_cond[a][b][c] = logsumexp_result.result + p_ij_cond[a * A_j + b];
        dv_p_ij_cond_signs[a][b][c] = logsumexp_result.sign;
      }
    }
  }

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("dv_p_ij_cond", dv_p_ij_cond, A_i_p_A_j*AA_ij);
  print_array_dbg_loc("dv_p_ij_cond_signs", consts->dv_p_ij_cond_signs, A_i_p_A_j*AA_ij);
  #endif

  initialize_array(consts->dw_p_ij_cond, log0, AA_ij * AA_ij_padded);
  initialize_array(consts->dw_p_ij_cond_signs, c_f1, AA_ij * AA_ij_padded);
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

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("dw_p_ij_cond", dw_p_ij_cond, AA_ij*AA_ij);
  print_array_dbg_loc("dw_p_ij_cond_signs", consts->dw_p_ij_cond_signs, AA_ij*AA_ij);
  #endif

  // p(.,b|a,.)
  c_float_t *p_ji_cond = consts->p_ji_cond;
  initialize_array(p_ji_cond, log0, AA_ij);
  for (int a = 0; a < A_i; a++) {
    for (int b = 0; b < A_j; b++) {
      c_float_t prob = (v[A_i + b] + w[a * A_j + b]) / loge_2;
      p_ji_cond[a * A_j + b] = prob;
      tmp_prob[b] = prob;
    }
    c_float_t normalization = logsumexpn(tmp_prob, A_j);
    for (int b = 0; b < A_j; b++) {
      p_ji_cond[a * A_j + b] -= normalization;
    }
  }

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("p_ji_cond", p_ji_cond, AA_ij);
  #endif

  initialize_array(consts->dv_p_ji_cond, log0, AA_ij * A_i_p_A_j_padded);
  initialize_array(consts->dv_p_ji_cond_signs, c_f1, AA_ij * A_i_p_A_j_padded);
  c_float_t (*dv_p_ji_cond)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ji_cond;
  c_float_t (*dv_p_ji_cond_signs)[A_j][A_i_p_A_j_padded] = (c_float_t (*)[A_j][A_i_p_A_j_padded]) consts->dv_p_ji_cond_signs;

  for (int d = 0; d < A_j; d++) {
    for (int a = 0; a < A_i; a++) {
      for (int b = 0; b < A_j; b++) {
        SignedLogExp logsumexp_result = signed_logsumexp2((b == d) ? c_f0: log0, c_f1, p_ji_cond[a * A_j + d], -c_f1);
        dv_p_ji_cond[a][b][A_i + d] = logsumexp_result.result + p_ji_cond[a * A_j + b];
        dv_p_ji_cond_signs[a][b][A_i + d] = logsumexp_result.sign;
      }
    }
  }

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("dv_p_ji_cond", dv_p_ji_cond, A_i_p_A_j*AA_ij);
  print_array_dbg_loc("dv_p_ji_cond_signs", consts->dv_p_ji_cond_signs, A_i_p_A_j*AA_ij);
  #endif

  c_float_t (*dw_p_ji_cond)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ji_cond;
  c_float_t (*dw_p_ji_cond_signs)[A_j][AA_ij_padded] = (c_float_t (*)[A_j][AA_ij_padded]) consts->dw_p_ji_cond_signs;

  initialize_array(consts->dw_p_ji_cond, log0, AA_ij * AA_ij_padded);
  initialize_array(consts->dw_p_ji_cond_signs, c_f1, AA_ij * AA_ij_padded);
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

  #ifdef DEBUG_PRINT
  print_array_dbg_loc("dw_p_ji_cond", dw_p_ji_cond, AA_ij*AA_ij);
  print_array_dbg_loc("dw_p_ji_cond_signs", consts->dw_p_ji_cond_signs, AA_ij*AA_ij);
  #endif
}

void initialize_logexpbuffer(LogExpBuffer* buffer, Constants* consts, size_t length) {
  int AA_ij = consts->AA_ij;
  // required size varies and is chosen large enough
  buffer->max1 = malloc_simd_farr(AA_ij * length * sizeof(c_float_t));
  buffer->max2 = malloc_simd_farr(AA_ij * length * sizeof(c_float_t));
  buffer->tmp_dim3 = malloc_simd_farr(length * sizeof(c_float_t));
}

void deinitialize_logexpbuffer(LogExpBuffer* buffer) {
  free(buffer->max1);
  free(buffer->max2);
  free(buffer->tmp_dim3);
}