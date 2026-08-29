// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// resonance_arch.h — the shape of the Resonance network.
//
// ---------------------------------------------------------------------------
// THE RESONANCE ARCHITECTURE
// ---------------------------------------------------------------------------
//
// Conventional NNUE evaluators anchor their feature set on the side-to-move's
// own king: every piece is encoded relative to where *my* king stands. That
// frame is excellent at describing shelter and structure, and comparatively
// blind to what a position looks like from the attacker's side of the board.
//
// Resonance evaluates every position twice, through two independent feature
// transformers that differ only in which king anchors the frame:
//
//   AEGIS — features bucketed by the *perspective side's own* king.
//           A defensive frame: "how safe and well-structured am I?"
//   LANCE — features bucketed by the *opponent's* king.
//           An offensive frame: "how much pressure is on their king?"
//
// Their scalar outputs are combined by a per-output-bucket learned gate:
//
//       eval  = (gate * AEGIS + (256 - gate) * LANCE) / 256
//
// The interesting part is not the blend but the residual:
//
//       sigma = |AEGIS - LANCE|
//
// Sigma measures how much the two frames disagree. It is large exactly where
// evaluation is least trustworthy: sharp positions where a defensive reading
// and an offensive reading diverge. The search consumes sigma directly to
// soften late move reductions in that high-disagreement tail, preserving a ply
// where a single scalar eval is least reliable.
//
// ---------------------------------------------------------------------------
// This header targets Resonance v10 only — the shipped champion. Earlier
// development carried #ifdef variants for v1..v9; those are not reproduced
// here. Carrying dead architectures into a clean tree buys nothing.
// ---------------------------------------------------------------------------
#pragma once
#include "types.h"

namespace resonance {

// ----------------------------- Feature space --------------------------------

// Perspective king squares collapse to 32 by horizontal mirroring, then into
// this many coarse buckets. Ten is a good trade: enough resolution to separate
// castled / central / advanced kings without exploding the transformer.
constexpr int KING_BUCKETS = 10;

// (bucket, piece class, square) with 12 piece classes (own P..K, their P..K).
constexpr int PIECE_CLASSES = 12;
constexpr int FEATURE_DIM   = KING_BUCKETS * PIECE_CLASSES * 64; // 7680

// Accumulator width, per perspective, per head.
//
// Two heads means two accumulators and two dense stacks per evaluation, so
// width buys quality at twice the usual price. Measured against the old
// hand-crafted evaluation, a 1024-wide pair cost about 190 Elo in pure speed
// on a laptop-class CPU and bought back only about 100 in quality — so the
// width is chosen to keep the network close to HCE in nodes per second, and
// the quality is bought with training data instead.
constexpr int L1 = 320;

// Output buckets, selected by total piece count: the network learns a distinct
// output head per game phase without needing a phase input.
constexpr int OUTPUT_BUCKETS = 8;

// ----------------------------- Dense stack ----------------------------------
//
// The accumulator is not fed forward as-is. Each perspective's L1 values are
// split in half and multiplied elementwise, giving L1/2 numbers per
// perspective and L1 in total. That halves the width of the first — and by far
// the most expensive — dense layer, and the product is a more useful primitive
// than the raw activation: it can only fire when *both* halves agree that
// something is present, which is exactly the conjunction a single linear layer
// cannot express.
constexpr int FC1_IN  = L1;           // 320
constexpr int FC1_OUT = 16;
constexpr int FC2_IN  = 2 * FC1_OUT;  // clipped + squared-clipped activations
constexpr int FC2_OUT = 64;
constexpr int FC3_IN  = FC2_OUT;

// The two frames. v10 accumulates each independently.
enum Head : int { HEAD_AEGIS = 0, HEAD_LANCE = 1, HEAD_NB = 2 };
constexpr int FRAME_NB = HEAD_NB;

// The policy head reuses both experts' final hidden activations, so no extra
// board network is evaluated during search.
constexpr int POLICY_CONTEXT_DIM = HEAD_NB * FC3_IN;
constexpr int POLICY_CLASSES     = 5 * 64 * 64;

// ----------------------------- Quantisation ---------------------------------

// Accumulator values live in int16 at this scale; activations are clipped to
// [0, 127] before the int8 dense layers.
constexpr int FT_SHIFT          = 9;
constexpr int FT_QUANT          = 127; // feature-transformer weight scale
constexpr int FC_QUANT          = 64;  // dense weight scale
constexpr int WEIGHT_SCALE_BITS = 6;   // log2(FC_QUANT)
constexpr int ACTIVATION_CLIP   = 127;

// Network output is produced at this scale and divided down to centipawn-ish
// internal units.
constexpr int OUTPUT_SCALE = 16;

// ----------------------------- File format ----------------------------------

constexpr uint32_t NET_MAGIC   = 0x414E5441u; // "ANTA"
constexpr uint32_t NET_VERSION = 0x000E0000u; // dual 320-lane frames, 64-wide heads

// A quick digest of the architecture; the loader refuses mismatched files.
// The mixed constants and their order must match the trainer's writer exactly
// or every existing .aa file stops loading.
constexpr uint32_t archHash() {
    uint32_t h = 2166136261u;
    auto mix = [&h](uint32_t x) {
        h ^= x;
        h *= 16777619u;
    };
    mix(uint32_t(KING_BUCKETS));
    mix(uint32_t(FEATURE_DIM));
    mix(uint32_t(L1));
    mix(uint32_t(OUTPUT_BUCKETS));
    mix(uint32_t(FC1_OUT));
    mix(uint32_t(FC2_OUT));
    mix(uint32_t(HEAD_NB));
    mix(0x44463330u); // "DF30": dual-frame 320-lane structured prune
    return h;
}

} // namespace resonance
