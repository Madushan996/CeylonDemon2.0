// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// eval.h — the evaluation entry point.
//
// There is no hand-crafted fallback. The 1.x line carried a tapered PeSTO
// evaluation for `Use NNUE=false`, but the Resonance network is embedded in
// the executable and always present, so the fallback had no reachable use
// case — and removing it took the last third-party table out of the tree.
//
// Evaluations are in Resonance's internal units, where a pawn is about 208.
#pragma once
#include "position.h"
#include "resonance_net.h"

// Static evaluation from the side to move's point of view. When `sigma` is
// given it receives the disagreement between the two frames, which the search
// spends on window width, reductions and pruning margins.
inline int evaluate(const Position& pos, int* sigma = nullptr) {
    if (!resonance::weightsLoaded) {
        if (sigma) *sigma = 0;
        return 0;
    }
    return resonance::evaluate(pos.acc, pos.byColor, pos.byPiece, pos.stm, sigma);
}
