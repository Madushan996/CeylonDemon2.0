// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// resonance_features.h — feature indexing for the two Resonance frames.
//
// Every feature is a (king bucket, piece class, square) triple, all three read
// from one perspective. Two things are folded out before indexing:
//
//   * vertical  — the board is flipped so the perspective side always plays up
//                 the board (relativeSquare);
//   * horizontal— if the anchoring king stands on files e-h, king and piece are
//                 both mirrored into files a-d, halving the king space to 32.
//
// The frames differ only in which king anchors them: AEGIS on the perspective
// side's own king, LANCE on the opponent's.
//
// NOTE ON NUMBERING: the trainer that produced the shipped networks numbers
// PAWN = 1, following the Stockfish NNUE convention, so its index needed
// `type_of(pc) - 1`. This engine numbers PAWN = 0 and uses the piece type
// directly. The resulting feature index is identical, which is what allows
// those networks to load unchanged; a differential test covers all
// 64 x 64 x 12 x 2 combinations. See NOTICE.md.
#pragma once
#include "resonance_arch.h"
#include "types.h"

namespace resonance {

// King-square buckets, indexed by the *relative, mirrored* king square.
// Ranks 1-2 get fine resolution across files a-d (that is where castling
// structure lives); the rest coarsen into rank pairs.
// clang-format off
constexpr uint8_t KING_BUCKET[64] = {
    0, 1, 2, 3,  3, 2, 1, 0,
    0, 1, 2, 3,  3, 2, 1, 0,
    4, 4, 5, 5,  5, 5, 4, 4,
    4, 4, 5, 5,  5, 5, 4, 4,
    6, 6, 7, 7,  7, 7, 6, 6,
    6, 6, 7, 7,  7, 7, 6, 6,
    8, 8, 9, 9,  9, 9, 8, 8,
    8, 8, 9, 9,  9, 9, 8, 8,
};
// clang-format on

// Square as seen from `persp`: black flips the board vertically.
inline int relativeSquare(int persp, int sq) { return sq ^ (persp * 56); }

// Mirror horizontally (a1 <-> h1).
inline int flipFile(int sq) { return sq ^ 7; }

// Does the anchor king stand on the king-side half, so everything must be
// mirrored into files a-d?
inline bool mirrored(int relKsq) { return fileOf(relKsq) >= 4; }

// Which king anchors head `h` when reading from `persp`.
//   AEGIS — the perspective side's own king   (defensive frame)
//   LANCE — the opposing king                 (offensive frame)
inline int anchorColor(int h, int persp) { return h == HEAD_AEGIS ? persp : persp ^ 1; }

// Feature index for a piece of type `pt` and colour `pc` standing on `sq`,
// read from `persp`, with the frame anchored on the king at `ksq`.
inline int featureIndex(int persp, int ksq, int pc, int pt, int sq) {
    int relK = relativeSquare(persp, ksq);
    int relS = relativeSquare(persp, sq);

    if (mirrored(relK)) {
        relK = flipFile(relK);
        relS = flipFile(relS);
    }

    const int bucket     = int(KING_BUCKET[relK]);
    const int pieceClass = (pc == persp ? 0 : 6) + pt;

    return (bucket * PIECE_CLASSES + pieceClass) * 64 + relS;
}

// Which king bucket a frame anchored at `ksq` reads from `persp`, and whether
// it is mirrored. The accumulator must be fully refreshed whenever either
// changes, because every feature index in the frame shifts.
inline int kingBucketOf(int persp, int ksq) {
    const int relK = relativeSquare(persp, ksq);
    return int(KING_BUCKET[mirrored(relK) ? flipFile(relK) : relK]);
}

inline bool kingMirroredFor(int persp, int ksq) {
    return mirrored(relativeSquare(persp, ksq));
}

// Total piece count decides which output head speaks for this position.
inline int outputBucket(int pieceCount) {
    const int b = (pieceCount - 1) / 4;
    return b < OUTPUT_BUCKETS - 1 ? b : OUTPUT_BUCKETS - 1;
}

} // namespace resonance
