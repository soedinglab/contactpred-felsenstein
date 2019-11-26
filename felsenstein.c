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
  leaf->data->Ln_ab[a*A + b] = 1;
}

void initialize_buffer(NodeBuffer* buffer) {
  buffer->Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*A);
  buffer->Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*A);

  buffer->dv_Ln = (c_float_t*) malloc(sizeof(c_float_t)*N_COL*A);
  buffer->dv_Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*N_COL*AA);
  buffer->dv_Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*N_COL*AA);

  buffer->dw_Ln = (c_float_t*) malloc(sizeof(c_float_t)*AA);
  buffer->dw_Ln_ia = (c_float_t*) malloc(sizeof(c_float_t)*AAA);
  buffer->dw_Ln_jb = (c_float_t*) malloc(sizeof(c_float_t)*AAA);
}

void deinitialize_buffer(NodeBuffer* buffer) {
  free(buffer->Ln_ia);
  free(buffer->Ln_jb);
  free(buffer->dv_Ln);
  free(buffer->dv_Ln_ia);
  free(buffer->dv_Ln_jb);
  free(buffer->dw_Ln);
  free(buffer->dw_Ln_ia);
  free(buffer->dw_Ln_jb);
}

void precompute_buffer(NodeBuffer* buffer, NodePrecomputation* data, Constants* consts){

  c_float_t *p_ab = consts->p_ab;
  c_float_t *dv_p_ab = consts->dv_p_ab;
  c_float_t *dw_p_ab = consts->dw_p_ab;

  c_float_t *p_ij_cond = consts->p_ij_cond;
  c_float_t *dv_p_ij_cond = consts->dv_p_ij_cond;
  c_float_t *dw_p_ij_cond = consts->dw_p_ij_cond;

  c_float_t *p_ji_cond = consts->p_ji_cond;
  c_float_t *dv_p_ji_cond = consts->dv_p_ji_cond;
  c_float_t *dw_p_ji_cond = consts->dw_p_ji_cond;

  // Nulling out buffer
  buffer->Ln = 0;

  memset(buffer->Ln_ia, 0, sizeof(c_float_t)*A);
  memset(buffer->Ln_jb, 0, sizeof(c_float_t)*A);

  memset(buffer->dv_Ln, 0, sizeof(c_float_t)*N_COL*A);
  memset(buffer->dv_Ln_ia, 0, sizeof(c_float_t)*N_COL*AA);
  memset(buffer->dv_Ln_jb, 0, sizeof(c_float_t)*N_COL*AA);

  memset(buffer->dw_Ln, 0, sizeof(c_float_t)*AA);
  memset(buffer->dw_Ln_ia, 0, sizeof(c_float_t)*AAA);
  memset(buffer->dw_Ln_jb, 0, sizeof(c_float_t)*AAA);

  for(int a = 0; a < A; a++) {
    for(int b = 0; b < A; b++) {
      buffer->Ln += data->Ln_ab[a*A + b] * p_ab[a*A + b];
    }
  }
  for(int a = 0; a < A; a++) {
    for(int d = 0; d < A; d++) {
      buffer->Ln_ia[a] += data->Ln_ab[a*A + d] * p_ji_cond[d*A + a];
    }
  }
  for(int b = 0; b < A; b++) {
    for(int c = 0; c < A; c++) {
      buffer->Ln_jb[b] += data->Ln_ab[c*A + b] * p_ij_cond[c*A + b];
    }
  }

  // derivatives
  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for(int c_p = 0; c_p < A; c_p++) {
        for(int d_p = 0; d_p < A; d_p++) {
          buffer->dv_Ln[l*A + c] += data->dv_Ln_ab[l*AAA + c*AA + c_p*A + d_p] * p_ab[c_p*A + d_p];
          buffer->dv_Ln[l*A + c] += data->Ln_ab[c_p*A + d_p] * dv_p_ab[l*AAA + c*AA + c_p*A + d_p];
        }
      }
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for(int c_p = 0; c_p < A; c_p++) {
        for(int d_p = 0; d_p < A; d_p++) {
          buffer->dw_Ln[c*A + d] += data->dw_Ln_ab[c*AAA + d*AA + c_p*A + d_p] * p_ab[c_p*A + d_p];
          buffer->dw_Ln[c*A + d] += data->Ln_ab[c_p*A + d_p] * dw_p_ab[c*AAA + d*AA + c_p*A + d_p];
        }
      }
    }
  }


  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for (int a = 0; a < A; a++) {
        for (int d_p = 0; d_p < A; d_p++) {
          buffer->dv_Ln_ia[l*AA + c*A + a] += data->dv_Ln_ab[l*AAA + c*AA + a*A + d_p] * p_ji_cond[d_p*A + a];
          buffer->dv_Ln_ia[l*AA + c*A + a] += data->Ln_ab[a*A + d_p] * dv_p_ji_cond[l*AAA + c*AA + d_p*A + a];
        }
      }
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for (int a = 0; a < A; a++) {
        for (int d_p = 0; d_p < A; d_p++) {
          buffer->dw_Ln_ia[c*AA + d*A + a] += data->dw_Ln_ab[c*AAA + d*AA + a*A + d_p] * p_ji_cond[d_p*A + a];
          buffer->dw_Ln_ia[c*AA + d*A + a] += data->Ln_ab[a*A + d_p] * dw_p_ji_cond[c*AAA + d*AA + d_p*A + a];
        }
      }
    }
  }


  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for (int b = 0; b < A; b++) {
        for (int c_p = 0; c_p < A; c_p++) {
          buffer->dv_Ln_jb[l*AA + c*A + b] += data->dv_Ln_ab[l*AAA + c*AA + c_p*A + b] * p_ij_cond[c_p*A + b];
          buffer->dv_Ln_jb[l*AA + c*A + b] += data->Ln_ab[c_p*A + b] * dv_p_ij_cond[l*AAA + c*AA + c_p*A + b];
        }
      }
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for (int b = 0; b < A; b++) {
        for (int c_p = 0; c_p < A; c_p++) {
          buffer->dw_Ln_jb[c*AA + d*A + b] += data->dw_Ln_ab[c*AAA + d*AA + c_p*A + b] * p_ij_cond[c_p*A + b];
          buffer->dw_Ln_jb[c*AA + d*A + b] += data->Ln_ab[c_p*A + b] * dw_p_ij_cond[c*AAA + d*AA + c_p*A + b];
        }
      }
    }
  }

}

void initialize_node(Node* node) {
  NodePrecomputation* data = malloc(sizeof(NodePrecomputation));
  data->Ln_ab = (c_float_t*) calloc(sizeof(c_float_t), AA);
  data->dv_Ln_ab = (c_float_t*) calloc(sizeof(c_float_t), N_COL*AAA);
  data->dw_Ln_ab = (c_float_t*) calloc(sizeof(c_float_t), AAAA);
  node->data = data;
}

void deinitialize_node(Node* node) {
  NodePrecomputation* data = node->data;
  free(data->Ln_ab);
  free(data->dv_Ln_ab);
  free(data->dw_Ln_ab);
}

void compute_Ln_branch(Node* node, c_float_t phi, NodeBuffer* buffer, Constants* consts, c_float_t* L_ab, c_float_t* dv_L_ab, c_float_t* dw_L_ab, int a, int b) {

  NodePrecomputation *child_data = node->data;
  NodeBuffer* child_buffer = buffer;
  *L_ab += (1 - phi) * (1 - phi) * child_buffer->Ln;
  *L_ab += (1 - phi) * phi * (child_buffer->Ln_ia[a] + child_buffer->Ln_jb[b]);
  *L_ab += phi * phi * (child_data->Ln_ab[a*A + b]);

  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      c_float_t deriv = 0;
      deriv += (1 - phi) * (1 - phi) * child_buffer->dv_Ln[l*A + c];
      deriv +=
        (1 - phi) * phi * (child_buffer->dv_Ln_ia[l*AA + c*A + a] + child_buffer->dv_Ln_jb[l*AA + c*A + b]);
      deriv += phi * phi * child_data->dv_Ln_ab[l*AAA + c*AA + a*A + b];
      dv_L_ab[l*A + c] = deriv;
    }
  }

  for (int c = 0; c < A; c++) {
    for (int d = 0; d < A; d++) {
      c_float_t deriv = 0;
      deriv += (1 - phi) * (1 - phi) * child_buffer->dw_Ln[c*A + d];
      deriv +=
        (1 - phi) * phi * (child_buffer->dw_Ln_ia[c*AA + d*A + a] + child_buffer->dw_Ln_jb[c*AA + d*A + b]);
      deriv += phi * phi * child_data->dw_Ln_ab[c*AAA + d*AA + a*A + b];
      dw_L_ab[c*A + d] = deriv;
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

  c_float_t* dv_left_Lab = (c_float_t *) malloc(sizeof(c_float_t *)*N_COL*A);
  c_float_t* dv_right_Lab = (c_float_t *) malloc(sizeof(c_float_t *)*N_COL*A);

  c_float_t* dw_left_Lab = (c_float_t *) malloc(sizeof(c_float_t *)*AA);
  c_float_t* dw_right_Lab = (c_float_t *) malloc(sizeof(c_float_t *)*AA);


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
      c_float_t left_Lab = 1;
      c_float_t right_Lab = 1;

      if (node->left != NULL) {
        left_Lab = 0;
        memset(dv_left_Lab, c_f0, N_COL*A);
        memset(dw_left_Lab, c_f0, AA);
        compute_Ln_branch(node->left, node->phi_left, buf->left, consts, &left_Lab, dv_left_Lab, dw_left_Lab, a, b);
      }
      if (node->right != NULL) {
        right_Lab = 0;
        memset(dv_right_Lab, c_f0, N_COL * A);
        memset(dw_right_Lab, c_f0, AA);
        compute_Ln_branch(node->right, node->phi_right, buf->right, consts, &right_Lab, dv_right_Lab, dw_right_Lab, a,
                          b);
      }

      c_float_t Ln_ab = left_Lab * right_Lab;

      NodePrecomputation* node_data = node->data;
      node_data->Ln_ab[a*A + b] = Ln_ab;

      // combine derivatives for left and right children
      for(int l = 0; l < N_COL; l++){
        for(int c = 0; c < A; c++) {
          node_data->dv_Ln_ab[l*AAA + c*AA + a*A + b] =
            dv_left_Lab[l*A + c] * right_Lab + left_Lab * dv_right_Lab[l*A + c];
        }
      }

      for (int c = 0; c < A; c++) {
        for (int d = 0; d < A; d++) {
          node_data->dw_Ln_ab[c*AAA + d*AA + a*A + b] =
            dw_left_Lab[c*A + d] * right_Lab + left_Lab * dw_right_Lab[c*A + d];
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
  c_float_t fx = 0;
  for(int a = 0; a < A; a++) {
    for(int b = 0; b < A; b++) {
      fx += root->data->Ln_ab[a*A + b] * aa_freqs[a] * aa_freqs[b];
    }
  }

  memset(grad, c_f0, (N_COL*A + A*A)*sizeof(c_float_t));
  for(int l = 0; l < N_COL; l++) {
    for(int c = 0; c < A; c++) {
      for(int a = 0; a < A; a++) {
        for(int b = 0; b < A; b++) {
          grad[l*A + c] += root->data->dv_Ln_ab[l*AAA + c*AA + a*A + b] * aa_freqs[a] * aa_freqs[b];
        }
      }
    }
  }
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for(int a = 0; a < A; a++) {
        for(int b = 0; b < A; b++) {
          grad[N_COL*A + c*A + d] += root->data->dw_Ln_ab[c*AAA + d*AA + a*A + b] * aa_freqs[a] * aa_freqs[b];
        }
      }
    }
  }
  return fx;
}

void initialize_constants(Constants* consts) {
  consts->p_ab = (c_float_t*) malloc(sizeof(c_float_t) * AA);
  consts->dv_p_ab = (c_float_t*) malloc(sizeof(c_float_t) * N_COL*AAA);
  consts->dw_p_ab = (c_float_t*) calloc(AAAA, sizeof(c_float_t));
  consts->p_ij_cond = (c_float_t*) calloc(AA, sizeof(c_float_t));
  consts->dv_p_ij_cond = (c_float_t*) calloc(N_COL*AAA, sizeof(c_float_t));
  consts->dw_p_ij_cond = (c_float_t*) calloc(AAAA, sizeof(c_float_t));
  consts->p_ji_cond = (c_float_t*) calloc(AA, sizeof(c_float_t));
  consts->dv_p_ji_cond = (c_float_t*) calloc(N_COL*AAA, sizeof(c_float_t));
  consts->dw_p_ji_cond = (c_float_t*) calloc(AAAA, sizeof(c_float_t));
}

void deinitialize_constants(Constants* consts) {
  free(consts->p_ab);
  free(consts->dv_p_ab);
  free(consts->dw_p_ab);
  free(consts->p_ij_cond);
  free(consts->dv_p_ij_cond);
  free(consts->dw_p_ij_cond);
  free(consts->p_ji_cond);
  free(consts->dv_p_ji_cond);
  free(consts->dw_p_ji_cond);
}

void precalculate_constants(Constants* consts, c_float_t* v, c_float_t* w) {

  // p_ab related precomputations
  c_float_t total_sum = 0;
  c_float_t* p_ab = consts->p_ab;
  memset(p_ab, c_f0, sizeof(c_float_t)*AA);
  for(int a = 0; a < A; a++) {
    for(int b = 0; b < A; b++) {
      p_ab[a*A + b] = exp(v[0*A + a] + v[1*A + b] + w[a*A + b] );
      total_sum += p_ab[a*A + b];
    }
  }
  for(int ab = 0; ab < A*A; ab++) {
    p_ab[ab] /= total_sum;
  }

  c_float_t pi_a[A] = {0};
  for(int a = 0; a < A; a++) {
    for(int b = 0; b < A; b++) {
      pi_a[a] += p_ab[a*A + b];
    }
  }
  c_float_t pj_b[A] = {0};
  for(int b = 0; b < A; b++) {
    for (int a = 0; a < A; a++) {
      pj_b[b] += p_ab[a * A + b];
    }
  }

  c_float_t* dv_p_ab = consts->dv_p_ab;
  memset(dv_p_ab, c_f0, sizeof(c_float_t)*N_COL*AAA);
  for(int c = 0; c < A; c++) {
    for (int a = 0; a < A; a++) {
      for (int b = 0; b < A; b++) {
        int ind = 0 * AAA + c * AA + a * A + b;
        dv_p_ab[ind] += (int) (a == c);
        dv_p_ab[ind] -= pi_a[c];
        dv_p_ab[ind] *= p_ab[a * A + b];
      }
    }
  }
  for(int d = 0; d < A; d++) {
    for (int a = 0; a < A; a++) {
      for (int b = 0; b < A; b++) {
        int ind = 1 * AAA + d * AA + a * A + b;
        dv_p_ab[ind] += (int) (b == d);
        dv_p_ab[ind] -= pj_b[d];
        dv_p_ab[ind] *= p_ab[a * A + b];
      }
    }
  }

  c_float_t* dw_p_ab = consts->dw_p_ab;
  memset(dw_p_ab, c_f0, sizeof(c_float_t)*AAAA);
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for(int a = 0; a < A; a++) {
        for(int b = 0; b < A; b++) {
          int ind = c*AAA + d*AA + a*A + b;
          dw_p_ab[ind] += (int) (a == c && b == d);
          dw_p_ab[ind] -= p_ab[c*A + d];
          dw_p_ab[ind] *= p_ab[a*A + b];
        }
      }
    }
  }

  // p(a,.|.,b)
  c_float_t* p_ij_cond = consts->p_ij_cond;
  memset(p_ij_cond, c_f0, sizeof(c_float_t)*AA);
  for(int b = 0; b < A; b++) {
    c_float_t ij_cond_sum = 0;
    for(int a = 0; a < A; a++) {
      c_float_t prob = exp(v[0*A+a] + w[a*A + b]);
      p_ij_cond[a*A + b] = prob;
      ij_cond_sum += prob;
    }
    for(int a = 0; a < A; a++) {
      p_ij_cond[a*A + b] /= ij_cond_sum;
    }
  }

  c_float_t* dv_p_ij_cond = consts->dv_p_ij_cond;
  memset(dv_p_ij_cond, c_f0, sizeof(c_float_t)*N_COL*AAA);
  for(int c = 0; c < A; c++) {
    for(int a = 0; a < A; a++) {
      for(int b = 0; b < A; b++) {
        int ind = 0*AAA + c*AA + b*A + a;
        dv_p_ij_cond[ind] += (int) (a == c);
        dv_p_ij_cond[ind] -= p_ij_cond[c*A + b];
        dv_p_ij_cond[ind] *= p_ij_cond[a*A + b];
      }
    }
  }

  c_float_t* dw_p_ij_cond = consts->dw_p_ij_cond;
  memset(dw_p_ij_cond, c_f0, sizeof(c_float_t)*AAAA);
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for(int a = 0; a < A; a++) {
        int b = d;
        int ind = c*AAA + d*AA + c*A + b;
        dw_p_ij_cond[ind] += (int) (a == c);
        dw_p_ij_cond[ind] -= p_ij_cond[c*A + d];
        dw_p_ij_cond[ind] *= p_ij_cond[a*A + b];
      }
    }
  }

  // p(.,b|a,.)
  c_float_t* p_ji_cond = consts->p_ji_cond;
  memset(p_ji_cond, c_f0, sizeof(c_float_t)*AA);
  for(int a = 0; a < A; a++) {
    c_float_t ji_cond_sum = 0;
    for(int b = 0; b < A; b++) {
      c_float_t prob = exp(v[1*A+b] + w[a*A + b]);
      p_ji_cond[b*A + a] = prob;
      ji_cond_sum += prob;
    }
    for(int b = 0; b < A; b++) {
      p_ji_cond[b*A + a] /= ji_cond_sum;
    }
  }

  c_float_t* dv_p_ji_cond = consts->dv_p_ji_cond;
  memset(dv_p_ji_cond, c_f0, sizeof(c_float_t)*N_COL*AAA);
  for(int d = 0; d < A; d++) {
    for(int a = 0; a < A; a++) {
      for(int b = 0; b < A; b++) {
        int ind = 1*AAA + d*AA + b*A + a;
        dv_p_ji_cond[ind] += (int) (b == d);
        dv_p_ji_cond[ind] -= p_ji_cond[d*A + a];
        dv_p_ji_cond[ind] *= p_ji_cond[b*A + a];
      }
    }
  }

  c_float_t* dw_p_ji_cond = consts->dw_p_ij_cond;
  memset(dw_p_ji_cond, c_f0, sizeof(c_float_t)*AAAA);
  for(int c = 0; c < A; c++) {
    for(int d = 0; d < A; d++) {
      for(int b = 0; b < A; b++) {
        int a = c;
        int ind = c*AAA + d*AA + b*A + a;
        dw_p_ji_cond[ind] += (int) (b == d);
        dw_p_ji_cond[ind] -= p_ji_cond[d*A + c];
        dw_p_ji_cond[ind] *= p_ji_cond[b*A + a];
      }
    }
  }
}