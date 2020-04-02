#ifndef FELSENSTEIN_SIMD_FUNCTIONS_H
#define FELSENSTEIN_SIMD_FUNCTIONS_H

#include <immintrin.h>
#include <float.h>
#include <math.h>
#include "simd.h"

#define simd_padded(N)  (N + (VECSIZE_DOUBLE-1))/VECSIZE_DOUBLE*VECSIZE_DOUBLE
#define malloc_simd_farr(x)  malloc_simd_double(x)

#ifndef C_FLOAT_T
#define C_FLOAT_T
typedef double c_float_t;
#endif

typedef struct LogExpBuffer {
  c_float_t* max1;
  c_float_t* max2;
  c_float_t* tmp_dim3;
} LogExpBuffer;

static inline simdf64 simdf64_pow2(simdf64 x) {

  /*
   * Approximates the pow2(x) of double precision numbers
   *
   * decomposes x:= y1 + y2, where y1 := floor(x) and y2 = 0.zzzzzzz.
   * Uses a polynomial approximation of f := pow2(y2) f: [0, 1[ -> [1, 2[ gives the mantisse of the result
   * Note: f(0) = 1 and f(1) = 2, therefore h = 1 and g = 2-a-b-c-d-e-f-g-1
   * Maximum deviation: 4.2e-9
   */

  const simdf64 c_1d = simdf64_set(1);
  const simdi64 c_1023l = simdf64_set(1023); // 1023 is the offset of the exponent in double representation
  const simdi64 c_mantissa_mask = simdi64_set(0xfffffffffffff); // the 52 bits of the double mantissa set to 1

  // 6th order polynomial coefficients
  const simdf64 poly_a = simdf64_set(0.0002187767014305746);
  const simdf64 poly_b = simdf64_set(0.0012388813954882880);
  const simdf64 poly_c = simdf64_set(0.0096843277474313091);
  const simdf64 poly_d = simdf64_set(0.0554806806423937746);
  const simdf64 poly_e = simdf64_set(0.2402303737183841825);
  const simdf64 poly_f = simdf64_set(0.6931469597948718420);

  // decompose x = y1 + y2, where y1 := floor(x) and y2 := 1.zzzzzzzzz
  simdf64 y1 = simdf64_floor(x);
  simdf64 y2 = simdf64_sub(x, y1);

  // calculate the polynomial approximation f(y2) ~ 2^y.
  // mant := f(y2) = ((((((a x y2 + b)*y2 + c)*y2 + d)*y2 + e)*y2 + f)*y2 + g)*y2 + h
  simdf64 mant;
  mant = simdf64_mul(y2, poly_a);
  mant = simdf64_add(poly_b, mant);
  mant = simdf64_mul(y2, mant);
  mant = simdf64_add(poly_c, mant);
  mant = simdf64_mul(y2, mant);
  mant = simdf64_add(poly_d, mant);
  mant = simdf64_mul(y2, mant);
  mant = simdf64_add(poly_e, mant);
  mant = simdf64_mul(y2, mant);
  mant = simdf64_add(poly_f, mant);
  mant = simdf64_mul(y2, mant);
  mant = simdf64_add(mant, c_1d);

  // assemble the double number by putting together mantissa and exponent
  simdi64 mantissa_long = simdi64_and((simdi64) mant, c_mantissa_mask); // zero out everything but the mantissa digits
  simdi64 exp_i64 = simdi64_f2i(simdf64_add(y1, c_1023l)); // double exponent is stored with an offset of 1023
  simdi64 shifted_exp = simdi64_slli(exp_i64, 52); // double mantissa has 52 bits
  x = (simdf64) simdi64_or(mantissa_long, shifted_exp); // join mantisse and exponent and obtain the final result 2^x

  // mask out edge cases
  // if x > DBL_MAX_EXP 2^x -> inf, if x < DBL_MIN_EXP 2^x -> 0
  const simdf64 c_max_exp = simdf64_set(DBL_MAX_EXP);
  const simdf64 c_min_exp = simdf64_set(DBL_MIN_EXP);
  const simdf64 c_max = simdf64_set(INFINITY);
  simdf64 c_max_mask = simdf64_cmp(x, c_max_exp, _CMP_GT_OS);
  const simdf64 c_min = simdf64_set(0);
  simdf64 c_min_mask = simdf64_cmp(x, c_min_exp, _CMP_LT_OS);
  x = simdf64_blendv(x, c_max, c_max_mask);
  x = simdf64_blendv(x, c_min, c_min_mask);

  return x;
}

static inline simdf64 simdf64_log2(simdf64 x) {

  /*
   * Approximates the log2(x) of double precision numbers
   *
   * Based on: log2[2^e * 1.m] = e + log2[1.m]
   * Uses a polynomial approximation of f := log2(x+1) the interval of [1;2] for the log2 of the mantissa 1.m
   * Note: f(0) = 0 and f(1) = 1, therefore j = 0 and i = 1-a-b-c-d-e-f-g-h
   * Maximum deviation: 1.3e-8
   */

  // define coefficients for fitted polynomial of x+1.
  const simdf64 poly_a = simdf64_set(0.00539574483271335);
  const simdf64 poly_b = simdf64_set(-0.033134075405641866);
  const simdf64 poly_c = simdf64_set(0.09571929135783046);
  const simdf64 poly_d = simdf64_set(-0.18043327446159182);
  const simdf64 poly_e = simdf64_set(0.26625227022774905);
  const simdf64 poly_f = simdf64_set(-0.3553426744739997);
  const simdf64 poly_g = simdf64_set(0.4801415033950581);
  const simdf64 poly_h = simdf64_set(-0.7212923532638644);
  const simdf64 poly_i = simdf64_set(1.4426935677917467);


  const simdi64 c_min_norm_pos = simdi64_set(0x0010000000000000);
  const simdi64 c_1023 = simdi64_set(0x3ff);
  const simdi64 c_mantissa_mask = simdi64_set(0xfffffffffffff);
  const simdi64 c_exp_1023 = simdi64_set(0x3ff0000000000000);

  simdf64 const_1d = simdf64_set(1);

  simdf64 R;
  simdf64 e;

  x = simdf64_max(x, c_min_norm_pos);  /* cut off denormalized stuff */

  // can be done with AVX2
  e = simdi64_srli((simdi64) x, 52);
  e = simdi64_sub(e, c_1023);

  x = (simdf64) simdi64_and((simdi64) x, c_mantissa_mask);  // zero out exponent
  x = (simdf64) simdi64_or((simdi64) x, c_exp_1023);         // set exponent to 1023 (1023 - 1023 = 0)

  x = simdf64_sub(x, const_1d);         // subtract one from mantisse
  R = simdf64_mul(x, poly_a);           // R = a*X
  R = simdf64_add(R, poly_b);           // R = a*X+b
  R = simdf64_mul(R, x);                // R = (a*X+b)*X
  R = simdf64_add(R, poly_c);           // R = (a*X+b)*X+c
  R = simdf64_mul(R, x);                // R = ((a*X+b)*X+c)*X
  R = simdf64_add(R, poly_d);           // R = ((a*X+b)*X+c)*X+d
  R = simdf64_mul(R, x);                // R = (((a*X+b)*X+c)*X+d)*X
  R = simdf64_add(R, poly_e);           // R = (((a*X+b)*X+c)*X+d)*X+e
  R = simdf64_mul(R, x);                // R = ((((a*X+b)*X+c)*X+d)*X+e)*X
  R = simdf64_add(R, poly_f);           // R = ((((a*X+b)*X+c)*X+d)*X+e)*X+f
  R = simdf64_mul(R, x);                // R = (((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X
  R = simdf64_add(R, poly_g);           // R = (((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g
  R = simdf64_mul(R, x);                // R = ((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X
  R = simdf64_add(R, poly_h);           // R = (((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)
  R = simdf64_mul(R, x);                // R = ((((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)*X)
  R = simdf64_add(R, poly_i);           // R = ((((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)*X)+i
  R = simdf64_mul(R, x);                // R = (((((((((a*X+b)*X+c)*X+d)*X+e)*X+f)*X+g)*X+h)*X)+i)*X ~ log2(1+X) !!
  R = simdf64_add(R, simdf64_cvtepi64(e));  // convert integer exponent to float and add to mantisse

 return R;
}

static inline void add_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for (int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64 y_chunk = simdf64_load(y + n);
    simdf64_store(out + n,  simdf64_add(x_chunk, y_chunk));
  }
}

static inline void add_constant(c_float_t* out, c_float_t* x, c_float_t constant, size_t N) {
  simdf64 const_chunk = simdf64_set(constant);
  for (int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64_store(out + n,  simdf64_add(x_chunk, const_chunk));
  }
}

static inline void sub_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for (int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64 y_chunk = simdf64_load(y + n);
    simdf64_store(out + n,  simdf64_sub(x_chunk, y_chunk));
  }
}

static inline void mul_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for (int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64 y_chunk = simdf64_load(y + n);
    simdf64_store(out + n,  simdf64_mul(x_chunk, y_chunk));
  }
}

static inline void pow2_array(c_float_t* out, c_float_t* x, size_t N) {
  for(int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64_store(out + n , simdf64_pow2(x_chunk));
  }
}

static inline void log2_array(c_float_t* out, c_float_t* x, size_t N) {
  for(int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_simd = simdf64_load(x + n);
    simdf64_store(out + n, simdf64_log2(x_simd));
  }
}

static inline void max_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for(int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64 y_chunk = simdf64_load(y + n);
    simdf64_store(out + n,  simdf64_max(x_chunk, y_chunk));
  }
}

static inline void sign_array(c_float_t* out, c_float_t* x, size_t N) {

  simdf64 ones = simdf64_set(1);
  simdf64 sign_mask = simdf64_set(-0.0);

  for(int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64 signs = simdf64_and(x_chunk, sign_mask);
    simdf64_store(out + n, simdf64_xor(ones, signs));
  }
}

static inline void abs_array(c_float_t* out, c_float_t* x, size_t N) {
  simdf64 mask = simdf64_set(-0.0);
  for(int n = 0; n < N; n+=VECSIZE_DOUBLE) {
    simdf64 x_chunk = simdf64_load(x + n);
    simdf64_store(out + n,  simdf64_andnot(mask, x_chunk));
  }
}

static inline void signedlogsumexp2_array(c_float_t* out, c_float_t* out_signs, c_float_t* x, c_float_t* x_signs, c_float_t* y, c_float_t* y_signs, size_t N) {
  c_float_t max[N];
  max_array(max, x, y, N);

  c_float_t exp_x[N];
  sub_array(exp_x, x, max, N);
  pow2_array(exp_x, exp_x, N);
  mul_array(exp_x, exp_x, x_signs, N);

  c_float_t exp_y[N];
  sub_array(exp_y, y, max, N);
  pow2_array(exp_y, exp_y, N);
  mul_array(exp_y, exp_y, y_signs, N);

  add_array(exp_x, exp_x, exp_y, N);
  sign_array(out_signs, exp_x, N);
  abs_array(exp_x, exp_x, N);

  log2_array(exp_x, exp_x, N);
  add_array(out, max, exp_x, N);
}

static inline void signedlogsumexp3_array(c_float_t* out, c_float_t* out_signs, c_float_t* x, c_float_t* x_signs, c_float_t* y, c_float_t* y_signs,
                                   c_float_t* z, c_float_t* z_signs, size_t N) {

  c_float_t max[N];
  max_array(max, x, y, N);
  max_array(max, max, z, N);

  c_float_t exp_x[N];
  sub_array(exp_x, x, max, N);
  pow2_array(exp_x, exp_x, N);
  mul_array(exp_x, exp_x, x_signs, N);

  c_float_t exp_y[N];
  sub_array(exp_y, y, max, N);
  pow2_array(exp_y, exp_y, N);
  mul_array(exp_y, exp_y, y_signs, N);

  c_float_t exp_z[N];
  sub_array(exp_z, z, max, N);
  pow2_array(exp_z, exp_z, N);
  mul_array(exp_z, exp_z, z_signs, N);

  add_array(exp_x, exp_x, exp_y, N);
  add_array(exp_x, exp_x, exp_z, N);
  sign_array(out_signs, exp_x, N);
  abs_array(exp_x, exp_x, N);

  log2_array(exp_x, exp_x, N);
  add_array(out, max, exp_x, N);
}

static inline void col_max_ax01(int dim1, int dim2, int dim3, c_float_t *max,
                                c_float_t (*x)[dim2][dim3], c_float_t(*x_add)[dim2]) {

  simdf64 min_chunk = simdf64_set(-DBL_MAX);
  for(int cd = 0; cd < dim3; cd += VECSIZE_DOUBLE) {
    simdf64_store(max + cd, min_chunk);
  }

  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {
      simdf64 const_chunk = simdf64_set(x_add[c_p][d_p]);
      for(int cd = 0; cd < dim3; cd+=VECSIZE_DOUBLE) {
        simdf64 x_chunk = simdf64_load(&x[c_p][d_p][cd]);
        simdf64 chunk = simdf64_add(x_chunk, const_chunk);
        simdf64 max_chunk = simdf64_load(max + cd);
        simdf64 new_max = simdf64_max(chunk, max_chunk);
        simdf64_store(max + cd, new_max);
      }
    }
  }
}

static inline void col_max_ax0(int dim1, int dim2, int dim3, c_float_t (*max)[dim3],
                               c_float_t (*x)[dim2][dim3], c_float_t (*x_add)[dim2]) {
  simdf64 min_chunk = simdf64_set(-DBL_MAX);
  for(int d_p = 0; d_p < dim2; d_p++) {
    for(int cd = 0; cd < dim3; cd += VECSIZE_DOUBLE) {
      simdf64_store(&max[d_p][cd], min_chunk);
    }
  }
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {
      simdf64 const_chunk = simdf64_set(x_add[c_p][d_p]);
      for(int cd = 0; cd < dim3; cd+=VECSIZE_DOUBLE) {
        simdf64 x_chunk = simdf64_load(&x[c_p][d_p][cd]);
        simdf64 chunk = simdf64_add(x_chunk, const_chunk);
        simdf64 max_chunk = simdf64_load(&max[d_p][cd]);
        simdf64 new_max = simdf64_max(chunk, max_chunk);
        simdf64_store(&max[d_p][cd], new_max);
      }
    }
  }
}

static inline void col_max_ax1(int dim1, int dim2, int dim3, c_float_t (*max)[dim3],
                               c_float_t (*x)[dim2][dim3], c_float_t (*x_add)[dim2]) {
  simdf64 min_chunk = simdf64_set(-DBL_MAX);
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int cd = 0; cd < dim3; cd += VECSIZE_DOUBLE) {
      simdf64_store(&max[c_p][cd], min_chunk);
    }
  }
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {
      simdf64 const_chunk = simdf64_set(x_add[c_p][d_p]);
      for(int cd = 0; cd < dim3; cd+=VECSIZE_DOUBLE) {
        simdf64 x_chunk = simdf64_load(&x[c_p][d_p][cd]);
        simdf64 chunk = simdf64_add(x_chunk, const_chunk);
        simdf64 max_chunk = simdf64_load(&max[c_p][cd]);
        simdf64 new_max = simdf64_max(chunk, max_chunk);
        simdf64_store(&max[c_p][cd], new_max);
      }
    }
  }
}

static inline void logsumexp_matrix_ax01(int dim1, int dim2, int dim3, c_float_t* res, c_float_t* res_signs,
                                         c_float_t (*x1)[dim2][dim3], c_float_t (*sign1)[dim2][dim3], c_float_t (*y1)[dim2],
                                         c_float_t (*x2)[dim2][dim3], c_float_t (*sign2)[dim2][dim3], c_float_t (*y2)[dim2],
  LogExpBuffer* buf) {

  simdf64 zero_chunk = simdf64_set(0.0);
  for(int cd = 0; cd < dim3; cd+= VECSIZE_DOUBLE) {
    simdf64_store(res + cd, zero_chunk);
  }
  c_float_t* max1 = buf->max1;
  c_float_t* max2 = buf->max2;

  col_max_ax01(dim1, dim2, dim3, max1, x1, y1);
  col_max_ax01(dim1, dim2, dim3, max2, x2, y2);
  max_array(max1, max1, max2, dim3);

  c_float_t* tmp = buf->tmp_dim3;

  for (int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {

      c_float_t x1_const = y1[c_p][d_p];
      add_constant(tmp, x1[c_p][d_p], x1_const, dim3);
      sub_array(tmp, tmp, max1, dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign1[c_p][d_p], dim3);
      add_array(res, res, tmp, dim3);

      c_float_t x2_const = y2[c_p][d_p];
      add_constant(tmp, x2[c_p][d_p], x2_const, dim3);
      sub_array(tmp, tmp, max1, dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign2[c_p][d_p], dim3);
      add_array(res, res, tmp, dim3);
    }
  }

  sign_array(res_signs, res, dim3);
  abs_array(res, res, dim3);
  log2_array(res, res, dim3);
  add_array(res, res, max1, dim3);
}

static inline void logsumexp_matrix_ax0(int dim1, int dim2, int dim3, c_float_t (*res)[dim3], c_float_t (*res_signs)[dim3],
                                        c_float_t (*x1)[dim2][dim3], c_float_t (*sign1)[dim2][dim3], c_float_t (*y1)[dim2],
                                        c_float_t (*x2)[dim2][dim3], c_float_t (*sign2)[dim2][dim3], c_float_t (*y2)[dim2],
  LogExpBuffer* buffer) {

  simdf64 zero_chunk = simdf64_set(0.0);
  for(int d_p = 0; d_p < dim2; d_p++) {
    for(int cd = 0; cd < dim3; cd+= VECSIZE_DOUBLE) {
      simdf64_store(&res[d_p][cd], zero_chunk);
    }
  }

  c_float_t (*max1)[dim3] = (c_float_t (*)[dim3]) buffer->max1;
  c_float_t (*max2)[dim3] = (c_float_t (*)[dim3]) buffer->max2;

  col_max_ax0(dim1, dim2, dim3, max1, x1, y1);
  col_max_ax0(dim1, dim2, dim3, max2, x2, y2);

  c_float_t* max1_lin = (c_float_t*) max1;
  c_float_t* max2_lin = (c_float_t*) max2;
  max_array(max1_lin, max1_lin, max2_lin, dim2*dim3);
  c_float_t* tmp = buffer->tmp_dim3;

  for (int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {
      add_constant(tmp, x1[c_p][d_p], y1[c_p][d_p], dim3);
      sub_array(tmp, tmp, max1[d_p], dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign1[c_p][d_p], dim3);
      add_array(res[d_p], res[d_p], tmp, dim3);

      add_constant(tmp, x2[c_p][d_p], y2[c_p][d_p], dim3);
      sub_array(tmp, tmp, max1[d_p], dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign2[c_p][d_p], dim3);
      add_array(res[d_p], res[d_p], tmp, dim3);
    }
  }

  for(int d_p = 0; d_p < dim2; d_p++) {
    sign_array(res_signs[d_p], res[d_p], dim3);
    abs_array(res[d_p], res[d_p], dim3);
    log2_array(res[d_p], res[d_p], dim3);
    add_array(res[d_p], res[d_p], max1[d_p], dim3);
  }
}

static inline void logsumexp_matrix_ax1(int dim1, int dim2, int dim3, c_float_t (*res)[dim3], c_float_t (*res_signs)[dim3],
                                        c_float_t (*x1)[dim2][dim3], c_float_t (*sign1)[dim2][dim3], c_float_t (*y1)[dim2],
                                        c_float_t (*x2)[dim2][dim3], c_float_t (*sign2)[dim2][dim3], c_float_t (*y2)[dim2],
                                        LogExpBuffer* buffer) {

  simdf64 zero_chunk = simdf64_set(0.0);
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int cd = 0; cd < dim3; cd+= VECSIZE_DOUBLE) {
      simdf64_store(&res[c_p][cd], zero_chunk);
    }
  }

  c_float_t (*max1)[dim3] = (c_float_t (*)[dim3]) buffer->max1;
  c_float_t (*max2)[dim3] = (c_float_t (*)[dim3]) buffer->max2;
  col_max_ax1(dim1, dim2, dim3, max1, x1, y1);
  col_max_ax1(dim1, dim2, dim3, max2, x2, y2);

  c_float_t* max1_lin = (c_float_t*) max1;
  c_float_t* max2_lin = (c_float_t*) max2;
  max_array(max1_lin, max1_lin, max2_lin, dim1*dim3);

  c_float_t* tmp = buffer->tmp_dim3;

  for (int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {

      c_float_t x1_const = y1[c_p][d_p];
      add_constant(tmp, x1[c_p][d_p], x1_const, dim3);
      sub_array(tmp, tmp, max1[c_p], dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign1[c_p][d_p], dim3);
      add_array(res[c_p], res[c_p], tmp, dim3);

      c_float_t x2_const = y2[c_p][d_p];
      add_constant(tmp, x2[c_p][d_p], x2_const, dim3);
      sub_array(tmp, tmp, max1[c_p], dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign2[c_p][d_p], dim3);
      add_array(res[c_p], res[c_p], tmp, dim3);
    }
  }

  for(int c_p = 0; c_p < dim1; c_p++) {
    sign_array(res_signs[c_p], res[c_p], dim3);
    abs_array(res[c_p], res[c_p], dim3);
    log2_array(res[c_p], res[c_p], dim3);
    add_array(res[c_p], res[c_p], max1[c_p], dim3);
  }
}


#endif //FELSENSTEIN_SIMD_FUNCTIONS_H
