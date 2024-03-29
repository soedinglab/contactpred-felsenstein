//
// Created by Christian Roth on 21/01/2020.
//

#ifndef FELSENSTEIN_SIMD_FUNCTIONS_H
#define FELSENSTEIN_SIMD_FUNCTIONS_H

#include "float.h"
#include <immintrin.h>
#include "simd.h"

#define simd_padded(N)  (N + (VECSIZE_FLOAT-1))/VECSIZE_FLOAT*VECSIZE_FLOAT
#define malloc_simd_farr(x)  malloc_simd_float(x)

#ifndef C_FLOAT_T
#define C_FLOAT_T
typedef float c_float_t;
#endif

typedef struct LogExpBuffer {
  c_float_t* max1;
  c_float_t* max2;
  c_float_t* tmp_dim3;
} LogExpBuffer;

static inline simdf32 simdf32_fpow2(simdf32 X) {

  /////////////////////////////////////////////////////////////////////////////////////
  // SIMD 2^x for four floats
  // Calculate float of 2pow(x) for four floats in parallel with SSE2
  // ATTENTION: need to compile with g++ -fno-strict-aliasing when using -O2 or -O3!!!
  // Relative deviation < 4.6E-6  (< 2.3E-7 with 5'th order polynomial)
  //
  // Internal representation of float number according to IEEE 754 (__m128 --> 4x):
  //   1bit sign, 8 bits exponent, 23 bits mantissa: seee eeee emmm mmmm mmmm mmmm mmmm mmmm
  //                                    0x4b400000 = 0100 1011 0100 0000 0000 0000 0000 0000
  //   In summary: x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeee-127)
  /////////////////////////////////////////////////////////////////////////////////////

  simdi32* xPtr = (simdi32*) &X;    // store address of float as pointer to int

  const simdf32 CONST32_05f       = simdf32_set(0.5f); // Initialize a vector (4x32) with 0.5f
  // (3 << 22) --> Initialize a large integer vector (shift left)
  const simdi32 CONST32_3i          = simdi32_set(3);
  const simdi32 CONST32_3shift22    = simdi32_slli(CONST32_3i, 22);
  const simdf32 CONST32_1f        = simdf32_set(1.0f);
  const simdf32 CONST32_FLTMAXEXP = simdf32_set(FLT_MAX_EXP);
  const simdf32 CONST32_FLTMAX    = simdf32_set(FLT_MAX);
  const simdf32 CONST32_FLTMINEXP = simdf32_set(FLT_MIN_EXP);
  // fifth order
  const simdf32 CONST32_A = simdf32_set(0.00187682f);
  const simdf32 CONST32_B = simdf32_set(0.00898898f);
  const simdf32 CONST32_C = simdf32_set(0.0558282f);
  const simdf32 CONST32_D = simdf32_set(0.240153f);
  const simdf32 CONST32_E = simdf32_set(0.693153f);

  simdf32 tx;
  simdi32 lx;
  simdf32 dx;
  simdf32 result    = simdf32_set(0.0f);
  simdf32 maskedMax = simdf32_set(0.0f);
  simdf32 maskedMin = simdf32_set(0.0f);

  // Check wheter one of the values is bigger or smaller than FLT_MIN_EXP or FLT_MAX_EXP
  // The correct FLT_MAX_EXP value is written to the right place
  maskedMax = simdf32_gt(X, CONST32_FLTMAXEXP);
  maskedMin = simdf32_gt(X, CONST32_FLTMINEXP);
  maskedMin = simdf32_xor(maskedMin, maskedMax);
  // If a value is bigger than FLT_MAX_EXP --> replace the later result with FLTMAX
  maskedMax = simdf32_and(CONST32_FLTMAX, simdf32_gt(X, CONST32_FLTMAXEXP));

  tx = simdf32_add((simdf32 ) CONST32_3shift22, simdf32_sub(X, CONST32_05f)); // temporary value for truncation: x-0.5 is added to a large integer (3<<22),
  // 3<<22 = (1.1bin)*2^23 = (1.1bin)*2^(150-127),
  // which, in internal bits, is written 0x4b400000 (since 10010110bin = 150)

  lx = simdf32_f2i(tx);                                       // integer value of x

  dx = simdf32_sub(X, simdi32_i2f(lx));                       // float remainder of x

  //   x = 1.0f + dx*(0.693153f             // polynomial apporoximation of 2^x for x in the range [0, 1]
  //            + dx*(0.240153f             // Gives relative deviation < 2.3E-7
  //            + dx*(0.0558282f            // Speed: 2.3E-8s
  //            + dx*(0.00898898f
  //            + dx* 0.00187682f ))));
  X = simdf32_mul(dx, CONST32_A);
  X = simdf32_add(CONST32_B, X);  // add constant B
  X = simdf32_mul(dx, X);
  X = simdf32_add(CONST32_C, X);  // add constant C
  X = simdf32_mul(dx, X);
  X = simdf32_add(CONST32_D, X);  // add constant D
  X = simdf32_mul(dx, X);
  X = simdf32_add(CONST32_E, X);  // add constant E
  X = simdf32_mul(dx, X);
  X = simdf32_add(X, CONST32_1f); // add 1.0f

  simdi32 lxExp = simdi32_slli(lx, 23); // add integer power of 2 to exponent

  *xPtr = simdi32_add(*xPtr, lxExp); // add integer power of 2 to exponent

  // Add all Values that are greater than min and less than max
  result = simdf32_and(maskedMin, X);
  // Add MAX_FLT values where entry values were > FLT_MAX_EXP
  result = simdf32_or(result, maskedMax);

  return result;
}

static inline simdf32 simdf32_flog2(simdf32 X) {

  // Fast SIMD log2 for four floats
  // Calculate integer of log2 for four floats in parallel with SSE2
  // Maximum deviation: +/- 2.1E-5
  // Run time: ~5.6ns on Intel core2 2.13GHz.
  // For a negative argument, nonsense is returned. Otherwise, when <1E-38, a value
  // close to -126 is returned and when >1.7E38, +128 is returned.
  // The function makes use of the representation of 4-byte floating point numbers:
  // seee eeee emmm mmmm mmmm mmmm mmmm mmmm
  // s is the sign, eee eee e gives the exponent + 127 (in hex: 0x7f).
  // The following 23 bits give the mantisse, the binary digits after the decimal
  // point:  x = (-1)^s * 1.mmmmmmmmmmmmmmmmmmmmmmm * 2^(eeeeeeee-127)
  // Therefore,  log2(x) = eeeeeeee-127 + log2(1.mmmmmm...)
  //                     = eeeeeeee-127 + log2(1+y),  where y = 0.mmmmmm...
  //                     ~ eeeeeeee-127 + ((a*y+b)*y+c)*y
  // The coefficients a, b  were determined by a least squares fit, and c=1-a-b to get 1 at y=1.
  // Lower/higher order polynomials may be used for faster or more precise calculation:
  // Order 1: log2(1+y) ~ y
  // Order 2: log2(1+y) = (a*y + 1-a)*y, a=-0.3427
  //  => max dev = +/- 8E-3, run time ~ 3.8ns
  // Order 3: log2(1+y) = ((a*y+b)*y + 1-a-b)*y, a=0.1564, b=-0.5773
  //  => max dev = +/- 1E-3, run time ~ 4.4ns
  // Order 4: log2(1+y) = (((a*y+b)*y+c)*y + 1-a-b-c)*y, a=-0.0803 b=0.3170 c=-0.6748
  //  => max dev = +/- 1.4E-4, run time ~ 5.0ns?
  // Order 5: log2(1+y) = ((((a*y+b)*y+c)*y+d)*y + 1-a-b-c-d)*y, a=0.0440047 b=-0.1903190 c=0.4123442 d=-0.7077702
  //  => max dev = +/- 2.1E-5, run time ~ 5.6ns?

  const simdi32 CONST32_0x7f = simdi32_set(0x7f);
  const simdi32 CONST32_0x7fffff = simdi32_set(0x7fffff);
  const simdi32 CONST32_0x3f800000 = simdi32_set(0x3f800000);
  const simdf32  CONST32_1f = simdf32_set(1.0);
  // const float a=0.1564, b=-0.5773, c=1.0-a-b;  // third order
  const float a=0.0440047f, b=-0.1903190f, c=0.4123442f, d=-0.7077702f, e=1.0-a-b-c-d; // fifth order
  const simdf32  CONST32_A = simdf32_set(a);
  const simdf32  CONST32_B = simdf32_set(b);
  const simdf32  CONST32_C = simdf32_set(c);
  const simdf32  CONST32_D = simdf32_set(d);
  const simdf32  CONST32_E = simdf32_set(e);
  simdi32 E; // exponents of X
  simdf32 R; //  result
  E = simdi32_srli((simdi32) X, 23);    // shift right by 23 bits to obtain exponent+127
  E = simdi32_sub(E, CONST32_0x7f);     // subtract 127 = 0x7f
  X = (simdf32) simdi_and((simdi32) X, CONST32_0x7fffff);  // mask out exponent => mantisse
  X = (simdf32) simdi_or ((simdi32) X, CONST32_0x3f800000); // set exponent to 127 (i.e., 0)
  X = simdf32_sub(X, CONST32_1f);          // subtract one from mantisse
  R = simdf32_mul(X, CONST32_A);           // R = a*X
  R = simdf32_add(R, CONST32_B);           // R = a*X+b
  R = simdf32_mul(R, X);                   // R = (a*X+b)*X
  R = simdf32_add(R, CONST32_C);           // R = (a*X+b)*X+c
  R = simdf32_mul(R, X);                   // R = ((a*X+b)*X+c)*X
  R = simdf32_add(R, CONST32_D);           // R = ((a*X+b)*X+c)*X+d
  R = simdf32_mul(R, X);                   // R = (((a*X+b)*X+c)*X+d)*X
  R = simdf32_add(R, CONST32_E);           // R = (((a*X+b)*X+c)*X+d)*X+e
  R = simdf32_mul(R, X);                   // R = ((((a*X+b)*X+c)*X+d)*X+e)*X ~ log2(1+X) !!
  R = simdf32_add(R, simdi32_i2f(E));  // convert integer exponent to float and add to mantisse
  return R;
}

static inline void add_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for (int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32 y_chunk = simdf32_load(y + n);
    simdf32_store(out + n,  simdf32_add(x_chunk, y_chunk));
  }
}

static inline void add_constant(c_float_t* out, c_float_t* x, c_float_t constant, size_t N) {
  simdf32 const_chunk = simdf32_set(constant);
  for (int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32_store(out + n,  simdf32_add(x_chunk, const_chunk));
  }
}

static inline void sub_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for (int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32 y_chunk = simdf32_load(y + n);
    simdf32_store(out + n,  simdf32_sub(x_chunk, y_chunk));
  }
}

static inline void mul_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for (int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32 y_chunk = simdf32_load(y + n);
    simdf32_store(out + n,  simdf32_mul(x_chunk, y_chunk));
  }
}

static inline void pow2_array(c_float_t* out, c_float_t* x, size_t N) {
  for(int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32_store(out + n , simdf32_fpow2(x_chunk));
  }
}

static inline void log2_array(c_float_t* out, c_float_t* x, size_t N) {
  for(int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_simd = simdf32_load(x + n);
    simdf32_store(out + n, simdf32_flog2(x_simd));
  }
}

static inline void max_array(c_float_t* out, c_float_t* x, c_float_t* y, size_t N) {
  for(int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32 y_chunk = simdf32_load(y + n);
    simdf32_store(out + n,  simdf32_max(x_chunk, y_chunk));
  }
}

static inline void sign_array(c_float_t* out, c_float_t* x, size_t N) {

  simdf32 ones = simdf32_set(1);
  simdf32 sign_mask = simdf32_set(-0.0);

  for(int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32 signs = simdf32_and(x_chunk, sign_mask);
    simdf32_store(out + n, simdf32_xor(ones, signs));
  }
}

static inline void abs_array(c_float_t* out, c_float_t* x, size_t N) {
  simdf32 mask = simdf32_set(-0.0);
  for(int n = 0; n < N; n+=VECSIZE_FLOAT) {
    simdf32 x_chunk = simdf32_load(x + n);
    simdf32_store(out + n,  simdf32_andnot(mask, x_chunk));
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

static inline void col_max_ax01(int dim1, int dim2, int dim3, float *max,
  float (*x)[dim2][dim3], float(*x_add)[dim2]) {

  simdf32 min_chunk = simdf32_set(-FLT_MAX);
  for(int cd = 0; cd < dim3; cd += VECSIZE_FLOAT) {
    simdf32_store(max + cd, min_chunk);
  }

  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {
      simdf32 const_chunk = simdf32_set(x_add[c_p][d_p]);
      for(int cd = 0; cd < dim3; cd+=VECSIZE_FLOAT) {
        simdf32 x_chunk = simdf32_load(&x[c_p][d_p][cd]);
        simdf32 chunk = simdf32_add(x_chunk, const_chunk);
        simdf32 max_chunk = simdf32_load(max + cd);
        simdf32 new_max = simdf32_max(chunk, max_chunk);
        simdf32_store(max + cd, new_max);
      }
    }
  }
}

static inline void col_max_ax0(int dim1, int dim2, int dim3, float (*max)[dim3],
  float (*x)[dim2][dim3], float (*x_add)[dim2]) {
  simdf32 min_chunk = simdf32_set(-FLT_MAX);
  for(int d_p = 0; d_p < dim2; d_p++) {
    for(int cd = 0; cd < dim3; cd += VECSIZE_FLOAT) {
      simdf32_store(&max[d_p][cd], min_chunk);
    }
  }
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {
      simdf32 const_chunk = simdf32_set(x_add[c_p][d_p]);
      for(int cd = 0; cd < dim3; cd+=VECSIZE_FLOAT) {
        simdf32 x_chunk = simdf32_load(&x[c_p][d_p][cd]);
        simdf32 chunk = simdf32_add(x_chunk, const_chunk);
        simdf32 max_chunk = simdf32_load(&max[d_p][cd]);
        simdf32 new_max = simdf32_max(chunk, max_chunk);
        simdf32_store(&max[d_p][cd], new_max);
      }
    }
  }
}

static inline void col_max_ax1(int dim1, int dim2, int dim3, float (*max)[dim3],
  float (*x)[dim2][dim3], float (*x_add)[dim2]) {
  simdf32 min_chunk = simdf32_set(-FLT_MAX);
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int cd = 0; cd < dim3; cd += VECSIZE_FLOAT) {
      simdf32_store(&max[c_p][cd], min_chunk);
    }
  }
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {
      simdf32 const_chunk = simdf32_set(x_add[c_p][d_p]);
      for(int cd = 0; cd < dim3; cd+=VECSIZE_FLOAT) {
        simdf32 x_chunk = simdf32_load(&x[c_p][d_p][cd]);
        simdf32 chunk = simdf32_add(x_chunk, const_chunk);
        simdf32 max_chunk = simdf32_load(&max[c_p][cd]);
        simdf32 new_max = simdf32_max(chunk, max_chunk);
        simdf32_store(&max[c_p][cd], new_max);
      }
    }
  }
}

static inline void logsumexp_matrix_ax01(int dim1, int dim2, int dim3, float* res, float* res_signs,
  float (*x1)[dim2][dim3], float (*sign1)[dim2][dim3], float (*y1)[dim2],
  float (*x2)[dim2][dim3], float (*sign2)[dim2][dim3], float (*y2)[dim2],
  LogExpBuffer* buf) {

  simdf32 zero_chunk = simdf32_set(0.0f);
  for(int cd = 0; cd < dim3; cd+= VECSIZE_FLOAT) {
    simdf32_store(res + cd, zero_chunk);
  }
  float* max1 = buf->max1;
  float* max2 = buf->max2;

  col_max_ax01(dim1, dim2, dim3, max1, x1, y1);
  col_max_ax01(dim1, dim2, dim3, max2, x2, y2);
  max_array(max1, max1, max2, dim3);

  float* tmp = buf->tmp_dim3;

  for (int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {

      float x1_const = y1[c_p][d_p];
      add_constant(tmp, x1[c_p][d_p], x1_const, dim3);
      sub_array(tmp, tmp, max1, dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign1[c_p][d_p], dim3);
      add_array(res, res, tmp, dim3);

      float x2_const = y2[c_p][d_p];
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

static inline void logsumexp_matrix_ax0(int dim1, int dim2, int dim3, float (*res)[dim3], float (*res_signs)[dim3],
  float (*x1)[dim2][dim3], float (*sign1)[dim2][dim3], float (*y1)[dim2],
  float (*x2)[dim2][dim3], float (*sign2)[dim2][dim3], float (*y2)[dim2],
  LogExpBuffer* buffer) {

  simdf32 zero_chunk = simdf32_set(0.0f);
  for(int d_p = 0; d_p < dim2; d_p++) {
    for(int cd = 0; cd < dim3; cd+= VECSIZE_FLOAT) {
      simdf32_store(&res[d_p][cd], zero_chunk);
    }
  }

  float (*max1)[dim3] = (float (*)[dim3]) buffer->max1;
  float (*max2)[dim3] = (float (*)[dim3]) buffer->max2;

  col_max_ax0(dim1, dim2, dim3, max1, x1, y1);
  col_max_ax0(dim1, dim2, dim3, max2, x2, y2);

  float* max1_lin = (float*) max1;
  float* max2_lin = (float*) max2;
  max_array(max1_lin, max1_lin, max2_lin, dim2*dim3);
  float* tmp = buffer->tmp_dim3;

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

static inline void logsumexp_matrix_ax1(int dim1, int dim2, int dim3, float (*res)[dim3], float (*res_signs)[dim3],
  float (*x1)[dim2][dim3], float (*sign1)[dim2][dim3], float (*y1)[dim2],
  float (*x2)[dim2][dim3], float (*sign2)[dim2][dim3], float (*y2)[dim2],
  LogExpBuffer* buffer) {

  simdf32 zero_chunk = simdf32_set(0.0f);
  for(int c_p = 0; c_p < dim1; c_p++) {
    for(int cd = 0; cd < dim3; cd+= VECSIZE_FLOAT) {
      simdf32_store(&res[c_p][cd], zero_chunk);
    }
  }

  float (*max1)[dim3] = (float (*)[dim3]) buffer->max1;
  float (*max2)[dim3] = (float (*)[dim3]) buffer->max2;
  col_max_ax1(dim1, dim2, dim3, max1, x1, y1);
  col_max_ax1(dim1, dim2, dim3, max2, x2, y2);

  float* max1_lin = (float*) max1;
  float* max2_lin = (float*) max2;
  max_array(max1_lin, max1_lin, max2_lin, dim1*dim3);

  float* tmp = buffer->tmp_dim3;

  for (int c_p = 0; c_p < dim1; c_p++) {
    for(int d_p = 0; d_p < dim2; d_p++) {

      float x1_const = y1[c_p][d_p];
      add_constant(tmp, x1[c_p][d_p], x1_const, dim3);
      sub_array(tmp, tmp, max1[c_p], dim3);
      pow2_array(tmp, tmp, dim3);
      mul_array(tmp, tmp, sign1[c_p][d_p], dim3);
      add_array(res[c_p], res[c_p], tmp, dim3);

      float x2_const = y2[c_p][d_p];
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
