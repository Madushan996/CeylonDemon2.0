// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// resonance_simd.h — vector kernels for the network's inner loops.
//
// Two code paths, selected at compile time: AVX2 and a portable scalar
// fallback. Both must produce bit-identical results, because the same network
// file has to evaluate identically on every build. Integer addition is
// associative, so the lane-parallel and sequential accumulation orders in
// affine() agree exactly; there is no floating point anywhere in the forward
// pass.
//
// An SSE2 path was tried and dropped: it only ever covered addWeights and
// subWeights — clippedRelu and pairwiseClipped already fell through to scalar
// without AVX2 — so it bought very little and doubled the bit-identity surface.
// AVX2 is the shipping target (i7-6600U, Skylake: AVX2 and fast
// BMI2, no AVX-512) and scalar covers portability.
//
// Alignment contract: `acc`, `in` and `out` buffers must be 32-byte aligned.
// Weight pointers may not be — they are owned by std::vector, whose allocation
// is not guaranteed to be 32-byte aligned (Windows' allocator commonly happens
// to align it, glibc may return only 16), so every weight load is unaligned.
#pragma once
#include <algorithm>
#include "types.h"

#if defined(__AVX2__) || defined(USE_AVX2)
    #define RESONANCE_AVX2 1
    #include <immintrin.h>
#endif

namespace resonance::simd {

#if defined(RESONANCE_AVX2)
inline int32_t horizontalSum(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                                _mm256_extracti128_si256(value, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
    return _mm_cvtsi128_si32(sum);
}
#endif

// ----------------------------------------------------------------------------
// Accumulator arithmetic: dst = src +/- weights, over n int16 lanes.
// ----------------------------------------------------------------------------

inline void addWeights(int16_t* dst, const int16_t* src, const int16_t* w, int n) {
#if defined(RESONANCE_AVX2)
    for (int i = 0; i < n; i += 16) {
        const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi16(a, b));
    }
#else
    for (int i = 0; i < n; i++) dst[i] = int16_t(src[i] + w[i]);
#endif
}

inline void subWeights(int16_t* dst, const int16_t* src, const int16_t* w, int n) {
#if defined(RESONANCE_AVX2)
    for (int i = 0; i < n; i += 16) {
        const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_sub_epi16(a, b));
    }
#else
    for (int i = 0; i < n; i++) dst[i] = int16_t(src[i] - w[i]);
#endif
}

inline void copy(int16_t* dst, const int16_t* src, int n) { std::copy(src, src + n, dst); }

// ----------------------------------------------------------------------------
// Clipped ReLU: int16 accumulator -> uint8 activation in [0, 127].
// n must be a multiple of 32 on the AVX2 path.
// ----------------------------------------------------------------------------

inline void clippedRelu(uint8_t* out, const int16_t* in, int n) {
#if defined(RESONANCE_AVX2)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i cap  = _mm256_set1_epi16(127);

    for (int i = 0; i < n; i += 32) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i + 16));

        a = _mm256_min_epi16(_mm256_max_epi16(a, zero), cap);
        b = _mm256_min_epi16(_mm256_max_epi16(b, zero), cap);

        // packus interleaves the two 128-bit halves; permute puts them back.
        const __m256i packed = _mm256_packus_epi16(a, b);
        _mm256_store_si256(reinterpret_cast<__m256i*>(out + i),
                           _mm256_permute4x64_epi64(packed, 0xD8));
    }
#else
    for (int i = 0; i < n; i++) out[i] = uint8_t(std::clamp(int(in[i]), 0, 127));
#endif
}

// ----------------------------------------------------------------------------
// Pairwise clipped product.
//
//   out[i] = (clamp(acc[i], 0, 127) * clamp(acc[i + half], 0, 127)) >> 7
//
// for i in [0, half). Halves the width handed to the first dense layer, and
// the product only fires when both halves agree that something is present —
// a conjunction a single linear layer cannot express.
// `half` must be a multiple of 32 on the AVX2 path. L1/2 = 160 satisfies this.
// ----------------------------------------------------------------------------

inline void pairwiseClipped(uint8_t* out, const int16_t* acc, int half) {
#if defined(RESONANCE_AVX2)
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i cap   = _mm256_set1_epi16(127);
    const __m256i round = _mm256_set1_epi16(64);

    for (int i = 0; i < half; i += 32) {
        __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + 16));
        __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + half + i));
        __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + half + i + 16));

        a0 = _mm256_min_epi16(_mm256_max_epi16(a0, zero), cap);
        a1 = _mm256_min_epi16(_mm256_max_epi16(a1, zero), cap);
        b0 = _mm256_min_epi16(_mm256_max_epi16(b0, zero), cap);
        b1 = _mm256_min_epi16(_mm256_max_epi16(b1, zero), cap);

        // Both operands are in [0, 127], so the product fits in int16 with room
        // to spare for the rounding term.
        const __m256i p0 =
            _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(a0, b0), round), 7);
        const __m256i p1 =
            _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(a1, b1), round), 7);

        const __m256i packed = _mm256_packus_epi16(p0, p1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(out + i),
                           _mm256_permute4x64_epi64(packed, 0xD8));
    }
#else
    for (int i = 0; i < half; i++) {
        const int a = std::clamp(int(acc[i]), 0, 127);
        const int b = std::clamp(int(acc[i + half]), 0, 127);
        out[i]      = uint8_t((a * b + 64) >> 7);
    }
#endif
}

// ----------------------------------------------------------------------------
// Affine transform: out[j] = bias[j] + sum_i weights[j][i] * in[i]
// with `in` uint8 and `weights` int8, accumulating into int32.
//
// inDim must be a multiple of 32. Weights are row-major, one contiguous row
// per output.
// ----------------------------------------------------------------------------

inline void affine(int32_t* out, const uint8_t* in, const int8_t* weights,
                   const int32_t* bias, int inDim, int outDim) {
#if defined(RESONANCE_AVX2)
    const __m256i ones = _mm256_set1_epi16(1);

    // FC2 always has exactly 32 input bytes. Keep that vector resident and
    // process four independent rows together rather than reloading it per row;
    // FC2 is 64 rows wide in v10.
    if (inDim == 32) {
        const __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(in));
        int j = 0;
        for (; j + 3 < outDim; j += 4) {
            const int8_t* row = weights + size_t(j) * 32;
            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row));
            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 32));
            const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 64));
            const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 96));
            out[j]     = bias[j]     + horizontalSum(_mm256_madd_epi16(_mm256_maddubs_epi16(x, w0), ones));
            out[j + 1] = bias[j + 1] + horizontalSum(_mm256_madd_epi16(_mm256_maddubs_epi16(x, w1), ones));
            out[j + 2] = bias[j + 2] + horizontalSum(_mm256_madd_epi16(_mm256_maddubs_epi16(x, w2), ones));
            out[j + 3] = bias[j + 3] + horizontalSum(_mm256_madd_epi16(_mm256_maddubs_epi16(x, w3), ones));
        }
        for (; j < outDim; j++) {
            const int8_t* row = weights + size_t(j) * 32;
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row));
            out[j] = bias[j] + horizontalSum(_mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones));
        }
        return;
    }

    // FC1 has many rows over the same comparatively wide activation vector.
    // Four independent accumulators let each 32-byte input block be loaded once
    // instead of once per row. Lane accumulation order is unchanged, so the
    // integer results are bit-identical to the one-row-at-a-time form.
    // This is the four-row path.
    int j = 0;
    for (; j + 3 < outDim; j += 4) {
        const int8_t* row0 = weights + size_t(j) * inDim;
        const int8_t* row1 = row0 + inDim;
        const int8_t* row2 = row1 + inDim;
        const int8_t* row3 = row2 + inDim;
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        for (int i = 0; i < inDim; i += 32) {
            const __m256i x  = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));
            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row0 + i));
            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row1 + i));
            const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row2 + i));
            const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row3 + i));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w0), ones));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w1), ones));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w2), ones));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w3), ones));
        }

        out[j]     = bias[j]     + horizontalSum(acc0);
        out[j + 1] = bias[j + 1] + horizontalSum(acc1);
        out[j + 2] = bias[j + 2] + horizontalSum(acc2);
        out[j + 3] = bias[j + 3] + horizontalSum(acc3);
    }

    for (; j < outDim; j++) {
        const int8_t* row = weights + size_t(j) * inDim;
        __m256i acc = _mm256_setzero_si256();
        for (int i = 0; i < inDim; i += 32) {
            const __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(in + i));
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones));
        }
        out[j] = bias[j] + horizontalSum(acc);
    }
#else
    for (int j = 0; j < outDim; j++) {
        const int8_t* row = weights + size_t(j) * inDim;
        int32_t acc = bias[j];
        for (int i = 0; i < inDim; i++) acc += int32_t(row[i]) * int32_t(in[i]);
        out[j] = acc;
    }
#endif
}

} // namespace resonance::simd
