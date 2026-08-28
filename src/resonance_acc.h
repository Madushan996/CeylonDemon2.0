// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// resonance_acc.h — the four accumulator slices and their update rules.
//
// A slice is one (head, perspective) pair: AEGIS/LANCE x white/black, four in
// all. Each is L1 = 320 int16 lanes.
//
// ---------------------------------------------------------------------------
// WHY A KING MOVE IS TWO DIFFERENT EVENTS
// ---------------------------------------------------------------------------
//
// Every feature index in a slice is computed relative to that slice's anchor
// king, so moving the anchor king can invalidate the whole slice at once. But
// the king is also an ordinary piece that occupies a square. A king move is
// therefore both:
//
//   * an anchor change, for the two slices that king anchors — AEGIS[itsColor]
//     and LANCE[theOtherColor] — but only if the bucket or the mirror side
//     actually changed. Those slices need a full refresh.
//   * an ordinary piece move, for the other two slices, which are anchored on
//     the *other* king and so keep all their indices. Those stay incremental.
//
// When the anchor king moves but stays inside the same bucket and on the same
// mirror side, every index in the slice is unchanged, so the move is purely
// incremental there too — and because the bucket is provably identical before
// and after, the pre-move and post-move king square give the same index. That
// is what lets the caller pass the post-move king square unconditionally on
// the incremental path.
//
// An earlier implementation computed accumulators lazily, walking a state
// chain on demand. This engine updates eagerly in make/unmake: no accumulator stack, no per-ply computed[]
// flags, no chain walk. The whole pre-move accumulator is saved into the Undo
// record, so unmake is one contiguous restore and is exact by construction.
#pragma once
#include <vector>
#include "resonance_arch.h"
#include "resonance_features.h"
#include "resonance_simd.h"
#include "types.h"

namespace resonance {

// Feature-transformer weights. Populated by the network loader.
struct FeatureWeights {
    alignas(64) int16_t ftBias[FRAME_NB][L1];
    std::vector<int16_t> ftWeight[FRAME_NB]; // FEATURE_DIM * L1, row-major by feature
};

inline FeatureWeights ftw;
inline bool weightsLoaded = false;

// The four slices for one position.
struct alignas(64) Accumulator {
    alignas(64) int16_t v[FRAME_NB][2][L1];
};

// Slices are identified by a bit index so a refresh set fits in a small mask.
constexpr int sliceBit(int h, int p) { return h * 2 + p; }

// The two slices anchored on `c`'s king: AEGIS[c] and LANCE[c ^ 1].
constexpr int anchoredSlices(int c) {
    return (1 << sliceBit(HEAD_AEGIS, c)) | (1 << sliceBit(HEAD_LANCE, c ^ 1));
}

// --------------------------------------------------------------------------
// Whole-slice rebuild from the board. O(pieces).
// --------------------------------------------------------------------------

inline void refreshSlice(Accumulator& a, int h, int p, const u64 byColor[2],
                         const u64 byPiece[6], int anchorKsq) {
    int16_t* dst = a.v[h][p];
    simd::copy(dst, ftw.ftBias[h], L1);

    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++) {
            u64 b = byColor[c] & byPiece[pt];
            while (b) {
                const int sq  = poplsb(b);
                const int idx = featureIndex(p, anchorKsq, c, pt, sq);
                simd::addWeights(dst, dst, ftw.ftWeight[h].data() + size_t(idx) * L1, L1);
            }
        }
}

inline void refreshAll(Accumulator& a, const u64 byColor[2], const u64 byPiece[6],
                       int whiteKsq, int blackKsq) {
    if (!weightsLoaded) return;
    for (int h = 0; h < FRAME_NB; h++)
        for (int p = 0; p < 2; p++) {
            const int anchor = anchorColor(h, p);
            refreshSlice(a, h, p, byColor, byPiece, anchor == WHITE ? whiteKsq : blackKsq);
        }
}

// --------------------------------------------------------------------------
// Incremental piece add/remove.
//
// `skipMask` names slices that are about to be refreshed wholesale; updating
// them incrementally would be wasted work, and for an anchor change it would
// be wrong, since their indices are computed against a king that has moved.
// --------------------------------------------------------------------------

inline void addPiece(Accumulator& a, int c, int pt, int sq, int whiteKsq, int blackKsq,
                     int skipMask = 0) {
    for (int h = 0; h < FRAME_NB; h++)
        for (int p = 0; p < 2; p++) {
            if (skipMask & (1 << sliceBit(h, p))) continue;
            const int anchor = anchorColor(h, p);
            const int idx = featureIndex(p, anchor == WHITE ? whiteKsq : blackKsq, c, pt, sq);
            simd::addWeights(a.v[h][p], a.v[h][p],
                             ftw.ftWeight[h].data() + size_t(idx) * L1, L1);
        }
}

inline void subPiece(Accumulator& a, int c, int pt, int sq, int whiteKsq, int blackKsq,
                     int skipMask = 0) {
    for (int h = 0; h < FRAME_NB; h++)
        for (int p = 0; p < 2; p++) {
            if (skipMask & (1 << sliceBit(h, p))) continue;
            const int anchor = anchorColor(h, p);
            const int idx = featureIndex(p, anchor == WHITE ? whiteKsq : blackKsq, c, pt, sq);
            simd::subWeights(a.v[h][p], a.v[h][p],
                             ftw.ftWeight[h].data() + size_t(idx) * L1, L1);
        }
}

// --------------------------------------------------------------------------
// Which slices a king move invalidates.
//
// Returns a slice mask. Empty when the king stayed inside its bucket and on
// its mirror side, which is the common case.
// --------------------------------------------------------------------------

inline int refreshMaskForKingMove(int kingColor, int fromSq, int toSq) {
    int mask = 0;
    for (int h = 0; h < FRAME_NB; h++)
        for (int p = 0; p < 2; p++) {
            if (anchorColor(h, p) != kingColor) continue;
            if (kingBucketOf(p, fromSq) != kingBucketOf(p, toSq)
                || kingMirroredFor(p, fromSq) != kingMirroredFor(p, toSq))
                mask |= 1 << sliceBit(h, p);
        }
    return mask;
}

} // namespace resonance
