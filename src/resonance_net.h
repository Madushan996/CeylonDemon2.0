// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// resonance_net.h — dense stack, blend gate, sigma, and the .aa loader.
//
// The forward pass, per head, per output bucket:
//
//   accumulator -> pairwise clipped product (L1 -> L1/2 per perspective,
//                  side to move first, so the net always sees "us, then them")
//               -> FC1  320 -> 16
//               -> clipped activation concatenated with its square (16 -> 32)
//               -> FC2   32 -> 64
//               -> clipped activation (64)
//               -> FC3   64 -> 1
//
// Both heads run it; the scalars are blended by a per-bucket learned gate, and
// their absolute difference is sigma.
//
// Every rescale rounds to nearest rather than truncating. An arithmetic shift
// alone biases every activation downwards and the bias compounds through the
// stack; adding half a unit first keeps the quantised network within a couple
// of internal units of the float one it was trained as.
//
// SCALE NOTE: these are the network's internal units, where a pawn is about
// 208 — not the ~90 the search margins were originally written against. The
// network defines the scale, so the search margins are rescaled to it (see
// ev() in search.h) rather than rescaling the network output.
#pragma once
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "resonance_acc.h"
#include "resonance_arch.h"
#include "resonance_embedded.h"
#include "resonance_features.h"
#include "resonance_simd.h"
#include "types.h"

namespace resonance {

// Evaluations are kept inside this band so they never collide with mate scores.
//
// Derived from this engine's own score constants, so it sits strictly inside
// the mate band and an evaluation can never be mistaken for a mate score.
//
// A bound borrowed from an engine with a different score scale would not be
// safe here: this engine uses MATE = 30000, MATE_BOUND = 29000 and
// INF = 31000, so any clamp above those would let an evaluation exceed INF and
// be reported as a mate score by scoreString() and shifted by ply in
// scoreToTT(). Deriving the bound locally makes that impossible by
// construction.
constexpr int EVAL_LIMIT = MATE_BOUND - 1; // 28999

// ----------------------------------------------------------------------------
// Dense weights, per head and per output bucket.
// ----------------------------------------------------------------------------

struct DenseWeights {
    alignas(64) int8_t  fc1Weight[HEAD_NB][OUTPUT_BUCKETS][FC1_OUT * FC1_IN];
    alignas(64) int32_t fc1Bias  [HEAD_NB][OUTPUT_BUCKETS][FC1_OUT];

    alignas(64) int8_t  fc2Weight[HEAD_NB][OUTPUT_BUCKETS][FC2_OUT * FC2_IN];
    alignas(64) int32_t fc2Bias  [HEAD_NB][OUTPUT_BUCKETS][FC2_OUT];

    alignas(64) int8_t  fc3Weight[HEAD_NB][OUTPUT_BUCKETS][FC3_IN];
    int32_t             fc3Bias  [HEAD_NB][OUTPUT_BUCKETS];

    // Blend gate, in 1/256ths of the AEGIS head, per output bucket.
    int16_t gate[OUTPUT_BUCKETS];
};

inline DenseWeights dw;
inline std::string loadedFrom;

// ----------------------------------------------------------------------------
// Bounds-checked cursor over the network blob. The same parser serves a file
// read into memory and the copy linked into the executable, so there is exactly
// one description of the layout to keep in step with the trainer.
// ----------------------------------------------------------------------------

class Reader {
public:
    Reader(const unsigned char* data, size_t size) : p(data), end(data + size) {}
    template <typename T> bool read(T* dst, size_t count) {
        const size_t bytes = count * sizeof(T);
        if (size_t(end - p) < bytes) return false;
        memcpy(dst, p, bytes);
        p += bytes;
        return true;
    }
private:
    const unsigned char* p;
    const unsigned char* end;
};

// ----------------------------------------------------------------------------
// File format (little-endian throughout, the only byte order targeted):
//
//   u32 magic "ANTA", u32 version, u32 archHash, u32 reserved, char[64] desc
//   per frame:            ftBias[L1], ftWeight[FEATURE_DIM * L1]
//   per head, per bucket: fc1 weight+bias, fc2 weight+bias, fc3 weight+bias
//   gate[OUTPUT_BUCKETS]
//
// The policy head lives in a separate .aap file and is not read here.
// ----------------------------------------------------------------------------

inline bool loadFromMemory(const unsigned char* data, size_t size, const std::string& name) {
    weightsLoaded = false;
    if (!data || !size) return false;

    Reader in(data, size);

    uint32_t magic = 0, version = 0, hash = 0, reserved = 0;
    char description[64] = {};

    if (!in.read(&magic, 1) || !in.read(&version, 1) || !in.read(&hash, 1)
        || !in.read(&reserved, 1) || !in.read(description, 64))
        return false;

    if (magic != NET_MAGIC) {
        fprintf(stderr, "CeylonDemon: '%s' is not a network file\n", name.c_str());
        return false;
    }
    if (version != NET_VERSION) {
        fprintf(stderr, "CeylonDemon: '%s' uses unsupported network version %08X\n",
                name.c_str(), version);
        return false;
    }
    if (hash != archHash()) {
        fprintf(stderr,
                "CeylonDemon: '%s' was built for a different architecture "
                "(file %08X, engine %08X)\n", name.c_str(), hash, archHash());
        return false;
    }

    for (int h = 0; h < FRAME_NB; h++) {
        if (!in.read(ftw.ftBias[h], L1)) return false;
        ftw.ftWeight[h].resize(size_t(FEATURE_DIM) * L1);
        if (!in.read(ftw.ftWeight[h].data(), size_t(FEATURE_DIM) * L1)) return false;
    }

    for (int h = 0; h < HEAD_NB; h++)
        for (int b = 0; b < OUTPUT_BUCKETS; b++) {
            if (!in.read(dw.fc1Weight[h][b], FC1_OUT * FC1_IN)) return false;
            if (!in.read(dw.fc1Bias[h][b], FC1_OUT)) return false;
            if (!in.read(dw.fc2Weight[h][b], FC2_OUT * FC2_IN)) return false;
            if (!in.read(dw.fc2Bias[h][b], FC2_OUT)) return false;
            if (!in.read(dw.fc3Weight[h][b], FC3_IN)) return false;
            if (!in.read(&dw.fc3Bias[h][b], 1)) return false;
        }

    if (!in.read(dw.gate, OUTPUT_BUCKETS)) return false;

    loadedFrom    = name;
    weightsLoaded = true;
    return true;
}

inline std::string netInfo() {
    return weightsLoaded ? loadedFrom : std::string("<none>");
}

inline bool loadFromFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<unsigned char> blob((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
    return loadFromMemory(blob.data(), blob.size(), path);
}

// The shipping build links the network into the executable, so a release
// stays a single self-contained file. Without it, fall back
// to finding the .aa on disk.
inline bool loadEmbedded() {
#if defined(CEYLON_EMBEDDED_NET)
    return loadFromMemory(ceylonNetStart, size_t(ceylonNetEnd - ceylonNetStart), "<built in>");
#else
    for (const char* p : {"ceylondemon.aa", "networks/ceylondemon.aa", "../ceylondemon.aa"})
        if (loadFromFile(p)) return true;
    return false;
#endif
}

// ----------------------------------------------------------------------------
// One head's forward pass. Returns the scalar in internal evaluation units.
// ----------------------------------------------------------------------------

inline int32_t propagate(const Accumulator& acc, int h, int us, int b,
                         uint8_t* policyContext = nullptr) {
    alignas(64) uint8_t activations[FC1_IN];
    alignas(64) int32_t fc1Out[FC1_OUT];
    alignas(64) uint8_t fc2In[FC2_IN];
    alignas(64) int32_t fc2Out[FC2_OUT];
    alignas(64) uint8_t fc3In[FC3_IN];

    // Side to move first, so the network always sees "us, then them". Each
    // perspective contributes L1/2 pairwise products rather than L1 raw
    // activations.
    const int frame = h; // v10 accumulates each frame independently
    simd::pairwiseClipped(activations, acc.v[frame][us], L1 / 2);
    simd::pairwiseClipped(activations + L1 / 2, acc.v[frame][us ^ 1], L1 / 2);

    simd::affine(fc1Out, activations, dw.fc1Weight[h][b], dw.fc1Bias[h][b], FC1_IN, FC1_OUT);

    constexpr int ROUND = 1 << (WEIGHT_SCALE_BITS - 1);

    // First half of the next layer's input is the clipped activation; second
    // half is its square. A squared branch lets the network express "how far
    // from neutral" without spending a whole extra layer on it.
    for (int i = 0; i < FC1_OUT; i++) {
        int y = (fc1Out[i] + ROUND) >> WEIGHT_SCALE_BITS;
        y = y < 0 ? 0 : (y > ACTIVATION_CLIP ? ACTIVATION_CLIP : y);
        fc2In[i]           = uint8_t(y);
        fc2In[FC1_OUT + i] = uint8_t((y * y + 64) >> 7);
    }

    simd::affine(fc2Out, fc2In, dw.fc2Weight[h][b], dw.fc2Bias[h][b], FC2_IN, FC2_OUT);

    for (int i = 0; i < FC2_OUT; i++) {
        int y = (fc2Out[i] + ROUND) >> WEIGHT_SCALE_BITS;
        fc3In[i] = uint8_t(y < 0 ? 0 : (y > ACTIVATION_CLIP ? ACTIVATION_CLIP : y));
    }

    if (policyContext) memcpy(policyContext, fc3In, FC3_IN);

    int32_t out = dw.fc3Bias[h][b];
    for (int i = 0; i < FC3_IN; i++) out += int32_t(dw.fc3Weight[h][b][i]) * int32_t(fc3In[i]);

    // The trainer scales the final layer so that dividing by the weight quant
    // lands directly in the engine's internal evaluation units. Integer
    // division truncates toward zero, so the rounding term carries the sign.
    return (out + (out >= 0 ? FC_QUANT / 2 : -FC_QUANT / 2)) / FC_QUANT;
}

// ----------------------------------------------------------------------------
// A small, exact-scope conversion guide for the elementary endings where one
// side has literally only its king and the other owns a rook or queen.
// Material already says the position is winning; this only breaks shallow
// search ties toward the standard mating plan: confine, then approach.
//
// PROVENANCE: the two constants below (64 and 16) are carried forward from
// earlier development and were not re-derived by measurement in this codebase.
// The pattern itself — an edge-distance term plus a king-proximity term — is
// the standard "mop-up" evaluation described in the general chess-programming
// literature and predates any one engine; the implementation here is written
// against this engine's own board accessors. Flagged in NOTICE.md rather than
// presented as original work.
// ----------------------------------------------------------------------------

constexpr int MOPUP_EDGE     = 64;
constexpr int MOPUP_APPROACH = 16;

inline int loneKingMopup(const u64 byColor[2], const u64 byPiece[6], int stm) {
    const auto bareKing = [&](int c) { return byColor[c] == (byColor[c] & byPiece[KING]); };

    int strong = WHITE, weak = BLACK;
    const u64 whiteHeavy = byColor[WHITE] & (byPiece[ROOK] | byPiece[QUEEN]);
    const u64 blackHeavy = byColor[BLACK] & (byPiece[ROOK] | byPiece[QUEEN]);

    if (bareKing(WHITE) && blackHeavy) {
        strong = BLACK; weak = WHITE;
    } else if (!(bareKing(BLACK) && whiteHeavy)) {
        return 0;
    }

    const int weakKing   = lsb(byColor[weak] & byPiece[KING]);
    const int strongKing = lsb(byColor[strong] & byPiece[KING]);

    const int wf = fileOf(weakKing), wr = rankOf(weakKing);
    int edgeDistance = wf < 7 - wf ? wf : 7 - wf;
    if (wr < edgeDistance) edgeDistance = wr;
    if (7 - wr < edgeDistance) edgeDistance = 7 - wr;

    // Chebyshev king distance.
    const int df = fileOf(strongKing) - wf, dr = rankOf(strongKing) - wr;
    const int af = df < 0 ? -df : df, ar = dr < 0 ? -dr : dr;
    const int kingDistance = af > ar ? af : ar;

    const int bonus = (3 - edgeDistance) * MOPUP_EDGE + (7 - kingDistance) * MOPUP_APPROACH;
    return stm == strong ? bonus : -bonus;
}

// ----------------------------------------------------------------------------
// Full evaluation: both heads, blended, with sigma reported.
// ----------------------------------------------------------------------------

inline int evaluate(const Accumulator& acc, const u64 byColor[2], const u64 byPiece[6],
                    int stm, int* sigma = nullptr) {
    const int b = outputBucket(popcount(byColor[WHITE] | byColor[BLACK]));

    const int32_t aegis = propagate(acc, HEAD_AEGIS, stm, b);
    const int32_t lance = propagate(acc, HEAD_LANCE, stm, b);

    // The residual between the defensive and offensive readings is the whole
    // point of running two heads: it is large exactly where a single scalar
    // evaluation is least trustworthy, and the search spends it on wider
    // windows and softer reductions.
    if (sigma) *sigma = aegis > lance ? aegis - lance : lance - aegis;

    int g = int(dw.gate[b]);
    g = g < 0 ? 0 : (g > 256 ? 256 : g);
    int32_t blended = (aegis * g + lance * (256 - g)) / 256;

    blended += loneKingMopup(byColor, byPiece, stm);

    return blended < -EVAL_LIMIT ? -EVAL_LIMIT : (blended > EVAL_LIMIT ? EVAL_LIMIT : blended);
}

} // namespace resonance
