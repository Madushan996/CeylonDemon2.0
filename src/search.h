// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake. GNU GPL v3+; see LICENSE.
// Built on the NARC Engine codebase by the same author (GPLv3).
//
// CeylonDemon — transposition table, search (Lazy SMP), time management
#pragma once
#include "eval.h"
#include <memory>
#include <thread>

// ----------------------------- Transposition table ---------------------------
// Shared across threads. Entries store key ^ data so a torn read/write under a
// data race is detected as a key mismatch (a miss) rather than corrupting play.
enum Bound : u8 { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    u64 key;  // zobrist ^ data
    u64 data; // packed: move(16) | score(16) | eval(16) | depth(8) | bound(8)
};

struct TT {
    static constexpr size_t CLUSTER_SIZE = 4;
    std::vector<TTEntry> table;
    u64 mask = 0;

    void resize(size_t mb) {
        size_t entries = mb * 1024 * 1024 / sizeof(TTEntry);
        size_t buckets = 1;
        while ((buckets * 2) * CLUSTER_SIZE <= entries) buckets *= 2;
        table.assign(buckets * CLUSTER_SIZE, TTEntry{});
        mask = buckets - 1;
    }
    void clear() { std::fill(table.begin(), table.end(), TTEntry{}); }

    bool probe(u64 key, Move& move, int& score, int& eval, int& depth, int& bound) const {
        size_t base = (key & mask) * CLUSTER_SIZE;
        for (size_t i = 0; i < CLUSTER_SIZE; i++) {
            const TTEntry& e = table[base + i];
            u64 d = e.data;
            if (d == 0 || (e.key ^ d) != key) continue;
            move  = Move(d & 0xFFFF);
            score = int(int16_t((d >> 16) & 0xFFFF));
            eval  = int(int16_t((d >> 32) & 0xFFFF));
            depth = int((d >> 48) & 0xFF);
            bound = int((d >> 56) & 0xFF);
            return true;
        }
        return false;
    }
    void store(u64 key, Move move, int score, int eval, int depth, int bound) {
        size_t base = (key & mask) * CLUSTER_SIZE;
        TTEntry* replace = &table[base];
        int replaceDepth = 256;
        for (size_t i = 0; i < CLUSTER_SIZE; i++) {
            TTEntry& candidate = table[base + i];
            u64 candidateData = candidate.data;
            if (candidateData == 0) { replace = &candidate; break; }
            if ((candidate.key ^ candidateData) == key) { replace = &candidate; break; }
            int candidateDepth = int((candidateData >> 48) & 0xFF);
            if (candidateDepth < replaceDepth) {
                replaceDepth = candidateDepth;
                replace = &candidate;
            }
        }
        TTEntry& e = *replace;
        u64 old = e.data;
        bool same = old != 0 && (e.key ^ old) == key;
        if (same) {
            int oldDepth = int((old >> 48) & 0xFF);
            if (bound != BOUND_EXACT && oldDepth > depth + 3) return;
            if (move == MOVE_NONE) move = Move(old & 0xFFFF); // keep old move
        }
        u64 d = u64(u16(move))
              | (u64(u16(int16_t(score))) << 16)
              | (u64(u16(int16_t(eval))) << 32)
              | (u64(u8(depth)) << 48)
              | (u64(u8(bound)) << 56);
        e.key = key ^ d;
        e.data = d;
    }
};

inline TT tt;

// ----------------------------- Search globals --------------------------------
struct SearchLimits {
    s64 softLimit = -1, hardLimit = -1; // ms
    int depthLimit = MAX_PLY - 1;
    u64 nodeLimit = 0;
    bool infinite = false;
};

inline std::atomic<bool> Stopped{false};
inline bool SilentSearch = false; // datagen: suppress info output
inline int Threads = 1;           // number of search threads (UCI "Threads")

// shared per-search settings, distributed to each thread at search start
inline SearchLimits searchLimits;
inline std::chrono::steady_clock::time_point searchStart;

// piece-square index helper for continuation history (piece 0-11 = color*6+type)
inline int psIdx(int pc12, int sq) { return pc12 * 64 + sq; }

struct SearchData {
    int  threadId = 0;
    bool isMain = true;
    u64 nodes = 0;
    int selDepth = 0;
    int rootDepth = 0;
    int completedDepth = 0;
    int lastScore = 0; // score of last completed iteration (for datagen)
    Move lastBest = MOVE_NONE; // best move from the last completed iteration

    Move killers[MAX_PLY + 2][2] = {};
    int  history[2][64][64] = {};
    Move counterMove[12][64] = {};
    // continuation history: previous move at 1, 2, and 4 plies back
    int16_t contHist[3][12 * 64][12 * 64] = {};
    // capture history: [attacker pc12][to][victim piece type]
    int16_t capHist[12][64][6] = {};

    // per-ply stack: what moved to where (for contHist/countermove); -1 = none/null
    int  stackPc12[MAX_PLY + 2];
    int  stackTo[MAX_PLY + 2];

    // per-thread copy of the game's position keys (repetition detection)
    u64 keyHist[1024 + MAX_PLY];
    int keyHistLen = 0;

    Move pvTable[MAX_PLY + 1][MAX_PLY + 1];
    int  pvLen[MAX_PLY + 1] = {};
    int  evalStack[MAX_PLY + 2] = {};

    s64 elapsed() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - searchStart).count();
    }
    void clearHistories() {
        memset(history, 0, sizeof(history));
        memset(counterMove, 0, sizeof(counterMove));
        memset(contHist, 0, sizeof(contHist));
        memset(capHist, 0, sizeof(capHist));
        memset(killers, 0, sizeof(killers));
    }
};

// thread pool (index 0 = main). Heap-allocated: each SearchData is ~2.4 MB.
inline std::vector<std::unique_ptr<SearchData>> threadData;

inline void initThreadData(int n) {
    while ((int)threadData.size() < n) threadData.push_back(std::make_unique<SearchData>());
    for (int i = 0; i < n; i++) { threadData[i]->threadId = i; threadData[i]->isMain = (i == 0); }
}
inline void clearAllHistories() {
    for (auto& td : threadData) td->clearHistories();
}
inline u64 totalNodes() {
    u64 n = 0;
    for (auto& td : threadData) n += td->nodes;
    return n;
}

// game-level position history (maintained by main.cpp); copied into each thread
inline u64 gameHist[1024 + MAX_PLY];
inline int gameHistLen = 0;

inline int lmrTable[64][64];
inline void initSearch() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++)
            lmrTable[d][m] = int(0.75 + std::log(double(d)) * std::log(double(m)) / 2.25);
    initThreadData(1);
}

inline void checkTime(SearchData& sd) {
    if (searchLimits.hardLimit >= 0 && sd.elapsed() >= searchLimits.hardLimit) Stopped = true;
    if (sd.isMain && searchLimits.nodeLimit && totalNodes() >= searchLimits.nodeLimit) Stopped = true;
}

inline bool isRepetitionOrFifty(const SearchData& sd, const Position& pos, int ply) {
    if (pos.halfmove >= 100) return true;
    int cur = sd.keyHistLen - 1 + ply;
    int limit = std::max(0, cur - pos.halfmove);
    for (int i = cur - 4; i >= limit; i -= 2)
        if (sd.keyHist[i] == pos.key) return true;
    return false;
}

inline bool insufficientMaterial(const Position& pos) {
    if (pos.byPiece[PAWN] | pos.byPiece[ROOK] | pos.byPiece[QUEEN]) return false;
    return popcount(pos.byPiece[KNIGHT] | pos.byPiece[BISHOP]) <= 1;
}

// ----------------------------- Move ordering ---------------------------------
constexpr int SCORE_TT      = 2000000000;
constexpr int SCORE_GOODCAP = 1000000000;
constexpr int SCORE_KILLER1 =  900000000;
constexpr int SCORE_KILLER2 =  899000000;
constexpr int SCORE_COUNTER =  898000000;
constexpr int SCORE_QUIET   =  500000000; // + combined history

// Resonance evaluates on a scale where a pawn is about 208. The pruning
// margins below were originally tuned against a net where a pawn was about 88,
// so every margin compared against a static eval or a search score has to be
// scaled by that ratio or it prunes roughly 2.4x too aggressively.
//
// NOT scaled: SEE thresholds, which work in the seeValue[] piece values, and
// history margins, which are in history units.
//
// This is a principled starting point, not a tuned set. Each of these wants
// re-measuring now that the evaluation underneath them has changed.
constexpr int ev(int cp) { return cp * 208 / 88; }
constexpr int SCORE_BADCAP  =  100000000;

// combined quiet history: main + continuation (1 and 2 plies back)
inline int quietHistScore(const SearchData& sd, const Position& pos, Move m, int ply) {
    int from = fromSq(m), to = toSq(m);
    int h = sd.history[pos.stm][from][to];
    int cur = psIdx(pos.stm * 6 + pos.board[from], to);
    if (ply >= 1 && sd.stackPc12[ply - 1] >= 0)
        h += sd.contHist[0][psIdx(sd.stackPc12[ply - 1], sd.stackTo[ply - 1])][cur];
    if (ply >= 2 && sd.stackPc12[ply - 2] >= 0)
        h += sd.contHist[1][psIdx(sd.stackPc12[ply - 2], sd.stackTo[ply - 2])][cur];
    if (ply >= 4 && sd.stackPc12[ply - 4] >= 0)
        h += sd.contHist[2][psIdx(sd.stackPc12[ply - 4], sd.stackTo[ply - 4])][cur] / 2;
    return h;
}

inline void scoreMoves(const SearchData& sd, const Position& pos, MoveList& ml,
                       Move ttMove, int ply) {
    Move counter = MOVE_NONE;
    if (ply >= 1 && sd.stackPc12[ply - 1] >= 0)
        counter = sd.counterMove[sd.stackPc12[ply - 1]][sd.stackTo[ply - 1]];

    for (int i = 0; i < ml.count; i++) {
        Move m = ml.list[i].move;
        if (m == ttMove) { ml.list[i].score = SCORE_TT; continue; }
        if (pos.isCapture(m)) {
            int victim = (flagOf(m) == ENPASSANT) ? PAWN : pos.board[toSq(m)];
            int att12 = pos.stm * 6 + pos.board[fromSq(m)];
            int mvvlva = 100 * seeValue[victim] - seeValue[pos.board[fromSq(m)]];
            int ch = sd.capHist[att12][toSq(m)][victim];
            int base = (see(pos, m) >= 0) ? SCORE_GOODCAP : SCORE_BADCAP;
            ml.list[i].score = base + mvvlva + ch * 4 + (flagOf(m) == PROMOTION ? 50000 : 0);
        } else if (flagOf(m) == PROMOTION) {
            ml.list[i].score = (promoOf(m) == QUEEN ? SCORE_GOODCAP - 1 : SCORE_BADCAP - 1);
        } else if (m == sd.killers[ply][0]) {
            ml.list[i].score = SCORE_KILLER1;
        } else if (m == sd.killers[ply][1]) {
            ml.list[i].score = SCORE_KILLER2;
        } else if (m == counter) {
            ml.list[i].score = SCORE_COUNTER;
        } else {
            ml.list[i].score = SCORE_QUIET + quietHistScore(sd, pos, m, ply);
        }
        // No root policy prior: Resonance's policy head lives in a separate
        // .aap file that this version does not load. It was measured off by
        // default in any case, unassisted search having scored better.
        if (ply == 0 && sd.threadId > 0) {
            // Helper threads seed the shared TT from different root branches
            // instead of duplicating the main thread's work in a slightly
            // different depth phase.
            u32 mix = u32(m) * 0x9E3779B9u ^ u32(sd.threadId) * 0x85EBCA6Bu;
            mix ^= mix >> 16;
            ml.list[i].score += int(mix % 200001u) - 100000;
        }
    }
}

inline Move pickMove(MoveList& ml, int idx) {
    int best = idx;
    for (int i = idx + 1; i < ml.count; i++)
        if (ml.list[i].score > ml.list[best].score) best = i;
    std::swap(ml.list[idx], ml.list[best]);
    return ml.list[idx].move;
}

inline void gravity(int& h, int bonus) {
    h += bonus - h * std::abs(bonus) / 16384;
}
inline void gravity16(int16_t& h, int bonus) {
    int v = h;
    v += bonus - v * std::abs(bonus) / 16384;
    h = int16_t(std::clamp(v, -30000, 30000));
}

// update all quiet-history tables for one move
inline void updateQuietHists(SearchData& sd, const Position& pos, Move m, int bonus, int ply) {
    gravity(sd.history[pos.stm][fromSq(m)][toSq(m)], bonus);
    int cur = psIdx(pos.stm * 6 + pos.board[fromSq(m)], toSq(m));
    if (ply >= 1 && sd.stackPc12[ply - 1] >= 0)
        gravity16(sd.contHist[0][psIdx(sd.stackPc12[ply - 1], sd.stackTo[ply - 1])][cur], bonus);
    if (ply >= 2 && sd.stackPc12[ply - 2] >= 0)
        gravity16(sd.contHist[1][psIdx(sd.stackPc12[ply - 2], sd.stackTo[ply - 2])][cur], bonus);
    if (ply >= 4 && sd.stackPc12[ply - 4] >= 0)
        gravity16(sd.contHist[2][psIdx(sd.stackPc12[ply - 4], sd.stackTo[ply - 4])][cur], bonus / 2);
}

// scores to/from TT need mate-distance adjustment
inline int scoreToTT(int s, int ply)   { return s >  MATE_BOUND ? s + ply : s < -MATE_BOUND ? s - ply : s; }
inline int scoreFromTT(int s, int ply) { return s >  MATE_BOUND ? s - ply : s < -MATE_BOUND ? s + ply : s; }

// ----------------------------- Quiescence ------------------------------------
inline int qsearch(SearchData& sd, Position& pos, int alpha, int beta, int ply) {
    const u64 timeMask = Threads > 1 ? 255 : 2047;
    if ((++sd.nodes & timeMask) == 0) checkTime(sd);
    if (Stopped) return 0;
    if (ply > sd.selDepth) sd.selDepth = ply;
    if (ply >= MAX_PLY - 1) return evaluate(pos);

    // Quiescence can contain long checked capture/evasion sequences. Record
    // those positions too, so repetitions and rule draws terminate with the
    // same semantics as the full-width search.
    sd.keyHist[sd.keyHistLen - 1 + ply] = pos.key;
    if (isRepetitionOrFifty(sd, pos, ply) || insufficientMaterial(pos))
        return VALUE_DRAW;

    bool isPV = (beta - alpha) > 1;

    Move ttMove = MOVE_NONE;
    int ttScore = 0, ttEval = INF, ttDepth = -1, ttBound = BOUND_NONE;
    bool ttHit = tt.probe(pos.key, ttMove, ttScore, ttEval, ttDepth, ttBound);
    if (ttHit) {
        ttScore = scoreFromTT(ttScore, ply);
        if (!isPV
            && (ttBound == BOUND_EXACT
                || (ttBound == BOUND_LOWER && ttScore >= beta)
                || (ttBound == BOUND_UPPER && ttScore <= alpha)))
            return ttScore;
    }

    bool inCheck = pos.inCheck();
    int best, rawEval = INF;

    if (inCheck) {
        best = -MATE + ply;
    } else {
        rawEval = (ttHit && ttEval != INF) ? ttEval : evaluate(pos);
        best = rawEval;
        if (best >= beta) return best;
        if (best > alpha) alpha = best;
    }

    MoveList ml;
    if (inCheck) genMoves<false>(pos, ml);
    else         genMoves<true>(pos, ml);
    scoreMoves(sd, pos, ml, ttMove, ply);

    int alphaOrig = alpha;
    Move bestMove = MOVE_NONE;
    Undo u;
    for (int i = 0; i < ml.count; i++) {
        Move m = pickMove(ml, i);
        if (!inCheck) {
            if (pos.isCapture(m) && see(pos, m) < 0) continue;
            int victim = (flagOf(m) == ENPASSANT) ? PAWN : pos.board[toSq(m)];
            int gain = (victim == NO_PIECE ? 0 : seeValue[victim])
                     + (flagOf(m) == PROMOTION ? seeValue[QUEEN] - seeValue[PAWN] : 0);
            // SEE material uses 100 units per pawn while Resonance uses about
            // 208. Compare delta gain and margin on the evaluator's scale.
            if (best + ev(gain + 150) <= alpha) continue;
        }
        int pc12 = pos.stm * 6 + pos.board[fromSq(m)];
        if (!pos.make(m, u)) continue;
        sd.stackPc12[ply] = pc12;
        sd.stackTo[ply] = toSq(m);
        int score = -qsearch(sd, pos, -beta, -alpha, ply + 1);
        pos.unmake(m, u);
        if (Stopped) return 0;
        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                bestMove = m;
                if (alpha >= beta) break;
            }
        }
    }

    int bound = best >= beta ? BOUND_LOWER
              : alpha != alphaOrig ? BOUND_EXACT : BOUND_UPPER;
    tt.store(pos.key, bestMove, scoreToTT(best, ply), rawEval, 0, bound);
    return best;
}

// ----------------------------- Main search -----------------------------------
inline int search(SearchData& sd, Position& pos, int alpha, int beta, int depth,
                  int ply, bool allowNull, Move excluded = MOVE_NONE,
                  int checksExtended = 0) {
    bool isPV = (beta - alpha) > 1;
    bool rootNode = (ply == 0);
    sd.pvLen[ply] = 0;

    if (depth <= 0) return qsearch(sd, pos, alpha, beta, ply);

    const u64 timeMask = Threads > 1 ? 255 : 2047;
    if ((++sd.nodes & timeMask) == 0) checkTime(sd);
    if (Stopped) return 0;
    if (ply >= MAX_PLY - 1) return evaluate(pos);

    sd.keyHist[sd.keyHistLen - 1 + ply] = pos.key;

    if (!rootNode) {
        if (isRepetitionOrFifty(sd, pos, ply) || insufficientMaterial(pos)) return VALUE_DRAW;
        alpha = std::max(alpha, -MATE + ply);
        beta  = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    Move ttMove = MOVE_NONE;
    int ttScore = 0, ttEval = INF, ttDepth = -1, ttBound = BOUND_NONE;
    bool ttHit = tt.probe(pos.key, ttMove, ttScore, ttEval, ttDepth, ttBound);
    if (ttHit) ttScore = scoreFromTT(ttScore, ply);
    if (ttHit && !isPV && excluded == MOVE_NONE && ttDepth >= depth) {
        if (ttBound == BOUND_EXACT
            || (ttBound == BOUND_LOWER && ttScore >= beta)
            || (ttBound == BOUND_UPPER && ttScore <= alpha))
            return ttScore;
    }

    bool inCheck = pos.inCheck();
    int staticEval, sigma = 0;
    if (inCheck) staticEval = -INF;
    else if (ttHit && ttEval != INF) staticEval = ttEval;
    else staticEval = evaluate(pos, &sigma);
    sd.evalStack[ply] = staticEval;
    bool improving = !inCheck && ply >= 2 && staticEval > sd.evalStack[ply - 2];

    if (!isPV && !inCheck && excluded == MOVE_NONE) {
        if (depth <= 3 && staticEval + ev(200) + ev(150) * depth <= alpha) {
            int rs = qsearch(sd, pos, alpha, alpha + 1, ply);
            if (rs <= alpha) return rs;
        }
        if (depth <= 8 && staticEval - ev(80) * depth + (improving ? ev(60) : 0) >= beta
            && std::abs(beta) < MATE_BOUND)
            return staticEval;

        if (allowNull && depth >= 3 && staticEval >= beta && pos.hasNonPawnMaterial()) {
            int R = 3 + depth / 3 + std::min(3, (staticEval - beta) / ev(200));
            Undo u;
            pos.makeNull(u);
            sd.stackPc12[ply] = -1;
            sd.stackTo[ply] = -1;
            int score = -search(sd, pos, -beta, -beta + 1, depth - 1 - R,
                                ply + 1, false, MOVE_NONE, checksExtended);
            pos.unmakeNull(u);
            if (Stopped) return 0;
            if (score >= beta) {
                // At high depth, verify the null-move cutoff from the real
                // position. This prevents a single optimistic null search
                // from pruning the node in zugzwang-like or tactically
                // unstable positions while keeping the common shallow path
                // unchanged.
                if (depth < 12)
                    return score > MATE_BOUND ? beta : score;
                else {
                    int verify = search(sd, pos, beta - 1, beta,
                                        depth - 1 - R, ply, false,
                                        MOVE_NONE, checksExtended);
                    if (Stopped) return 0;
                    if (verify >= beta)
                        return score > MATE_BOUND ? beta : score;
                }
            }
        }

        if (depth >= 5 && std::abs(beta) < MATE_BOUND) {
            int rBeta = std::min(beta + 180, MATE_BOUND - 1);
            MoveList pcml;
            genMoves<true>(pos, pcml);
            scoreMoves(sd, pos, pcml, ttMove, ply);
            Undo u;
            int tried = 0;
            for (int i = 0; i < pcml.count && tried < 6; i++) {
                Move m = pickMove(pcml, i);
                if (!pos.isCapture(m) || see(pos, m) < rBeta - staticEval) continue;
                int pc12 = pos.stm * 6 + pos.board[fromSq(m)];
                if (!pos.make(m, u)) continue;
                tried++;
                sd.stackPc12[ply] = pc12;
                sd.stackTo[ply] = toSq(m);
                int s = -qsearch(sd, pos, -rBeta, -rBeta + 1, ply + 1);
                if (s >= rBeta)
                    s = -search(sd, pos, -rBeta, -rBeta + 1, depth - 4,
                                ply + 1, true, MOVE_NONE, checksExtended);
                pos.unmake(m, u);
                if (Stopped) return 0;
                if (s >= rBeta) return s;
            }
        }
    }

    if (depth >= 4 && ttMove == MOVE_NONE) depth--;

    int singularExt = 0;
    if (!rootNode && excluded == MOVE_NONE && depth >= 8
        && ttMove != MOVE_NONE && ttHit && (ttBound & BOUND_LOWER)
        && ttDepth >= depth - 3 && std::abs(ttScore) < MATE_BOUND) {
        int sBeta = ttScore - ev(2) * depth;
        int sScore = search(sd, pos, sBeta - 1, sBeta, (depth - 1) / 2,
                            ply, false, ttMove, checksExtended);
        if (Stopped) return 0;
        if (sScore < sBeta) singularExt = 1;
        else if (sBeta >= beta) return sBeta;
    }

    MoveList ml;
    genMoves<false>(pos, ml);
    const Move orderingTT = rootNode && sd.threadId > 0 ? MOVE_NONE : ttMove;
    scoreMoves(sd, pos, ml, orderingTT, ply);

    int best = -INF;
    Move bestMove = MOVE_NONE;
    int legalMoves = 0, quietsTried = 0;
    Move quietList[128], capList[64];
    int quietCount = 0, capCount = 0;
    Undo u;

    for (int i = 0; i < ml.count; i++) {
        Move m = pickMove(ml, i);
        if (m == excluded) continue;
        bool isQuiet = !pos.isCapture(m) && flagOf(m) != PROMOTION;

        if (!rootNode && best > -MATE_BOUND) {
            if (isQuiet) {
                int lmpLimit = improving ? 4 + depth * depth : 2 + depth * depth / 2;
                if (depth <= 8 && quietsTried >= lmpLimit) continue;
                if (depth <= 4 && !inCheck && staticEval + ev(150) + ev(120) * depth <= alpha) continue;
                if (depth <= 4 && quietHistScore(sd, pos, m, ply) < -4000 * depth) continue;
            } else if (depth <= 8 && pos.isCapture(m) && see(pos, m) < -150 * depth) {
                continue;
            }
        }

        int histScore = isQuiet ? quietHistScore(sd, pos, m, ply) : 0;
        int pc12 = pos.stm * 6 + pos.board[fromSq(m)];
        if (!pos.make(m, u)) continue;
        legalMoves++;
        if (isQuiet) {
            quietsTried++;
            if (quietCount < 128) quietList[quietCount++] = m;
        } else if (pos.isCapture(m) || flagOf(m) == ENPASSANT) {
            if (capCount < 64) capList[capCount++] = m;
        }
        sd.stackPc12[ply] = pc12;
        sd.stackTo[ply] = toSq(m);

        bool givesCheck = pos.inCheck();
        // Resolve forcing checks a little beyond the nominal frontier, but
        // cap the allowance per line so checking cycles cannot explode.
        int checkExt = givesCheck && depth <= 8 && checksExtended < 2 ? 1 : 0;
        int ext = std::max((m == ttMove) ? singularExt : 0, checkExt);
        int childChecksExtended = checksExtended + checkExt;
        int newDepth = depth - 1 + ext;
        int score;

        if (legalMoves == 1) {
            score = -search(sd, pos, -beta, -alpha, newDepth, ply + 1, true,
                            MOVE_NONE, childChecksExtended);
        } else {
            int r = 0;
            if (depth >= 3 && legalMoves > 2 + 2 * isPV && isQuiet && !givesCheck) {
                r = lmrTable[std::min(depth, 63)][std::min(legalMoves, 63)];
                r -= isPV;
                r -= improving;
                if (m == sd.killers[ply][0] || m == sd.killers[ply][1]) r--;
                r -= std::clamp(histScore / 8192, -2, 2);
                // The two Resonance heads disagree most in tactically unstable
                // positions. Preserve one extra ply for only the high-sigma
                // tail, and only when this node required a fresh evaluation.
                if (sigma >= ev(100)) r--;
                // In materially decisive low-piece endings, quiet king and
                // pawn moves often carry the conversion plan. Search them one
                // ply less aggressively reduced; the small branching factor
                // keeps the cost bounded while reducing horizon-driven draws.
                if (!inCheck && popcount(pos.occupied()) <= 10
                    && std::abs(staticEval) >= 400)
                    r--;
                r = std::clamp(r, 0, newDepth - 1);
            }
            score = -search(sd, pos, -alpha - 1, -alpha, newDepth - r,
                            ply + 1, true, MOVE_NONE, childChecksExtended);
            if (score > alpha && r > 0)
                score = -search(sd, pos, -alpha - 1, -alpha, newDepth,
                                ply + 1, true, MOVE_NONE, childChecksExtended);
            if (score > alpha && score < beta)
                score = -search(sd, pos, -beta, -alpha, newDepth,
                                ply + 1, true, MOVE_NONE, childChecksExtended);
        }

        pos.unmake(m, u);
        if (Stopped) return 0;

        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                bestMove = m;
                if (isPV) {
                    sd.pvTable[ply][0] = m;
                    memcpy(&sd.pvTable[ply][1], &sd.pvTable[ply + 1][0],
                           sd.pvLen[ply + 1] * sizeof(Move));
                    sd.pvLen[ply] = sd.pvLen[ply + 1] + 1;
                }
                if (alpha >= beta) {
                    int bonus = std::min(2000, 16 * depth * depth);
                    if (isQuiet) {
                        if (sd.killers[ply][0] != m) {
                            sd.killers[ply][1] = sd.killers[ply][0];
                            sd.killers[ply][0] = m;
                        }
                        if (ply >= 1 && sd.stackPc12[ply - 1] >= 0)
                            sd.counterMove[sd.stackPc12[ply - 1]][sd.stackTo[ply - 1]] = m;
                        updateQuietHists(sd, pos, m, bonus, ply);
                        for (int q = 0; q < quietCount - 1; q++)
                            updateQuietHists(sd, pos, quietList[q], -bonus, ply);
                    } else if (pos.isCapture(m)) {
                        int victim = (flagOf(m) == ENPASSANT) ? PAWN : pos.board[toSq(m)];
                        gravity16(sd.capHist[pc12][toSq(m)][victim], bonus);
                    }
                    for (int q = 0; q < capCount; q++) {
                        Move cm = capList[q];
                        if (cm == m) continue;
                        int v = (flagOf(cm) == ENPASSANT) ? PAWN : pos.board[toSq(cm)];
                        if (v == NO_PIECE) continue;
                        int a12 = pos.stm * 6 + pos.board[fromSq(cm)];
                        gravity16(sd.capHist[a12][toSq(cm)][v], -bonus);
                    }
                    break;
                }
            }
        }
    }

    if (legalMoves == 0) {
        if (excluded != MOVE_NONE) return alpha;
        return inCheck ? -MATE + ply : VALUE_DRAW;
    }

    if (excluded == MOVE_NONE) {
        int bound = best >= beta ? BOUND_LOWER
                  : bestMove != MOVE_NONE ? BOUND_EXACT
                  : BOUND_UPPER;
        tt.store(pos.key, bestMove, scoreToTT(best, ply),
                 inCheck ? INF : sd.evalStack[ply], depth, bound);
    }

    return best;
}

// ----------------------------- Iterative deepening ---------------------------
inline std::string scoreString(int score) {
    std::ostringstream os;
    if (std::abs(score) > MATE_BOUND) {
        int mateIn = (MATE - std::abs(score) + 1) / 2;
        os << "mate " << (score > 0 ? mateIn : -mateIn);
    } else {
        os << "cp " << score;
    }
    return os.str();
}

// One thread's iterative-deepening loop. Helpers (isMain == false) neither print
// nor manage time; they desync via a per-thread start-depth offset and share the
// TT, which is the whole Lazy SMP mechanism.
inline void iterativeDeepening(SearchData& sd, Position pos) {
    sd.nodes = 0;
    sd.selDepth = 0;
    sd.completedDepth = 0;
    sd.lastBest = MOVE_NONE;
    memset(sd.killers, 0, sizeof(sd.killers));
    memset(sd.pvLen, 0, sizeof(sd.pvLen));
    for (int i = 0; i < MAX_PLY + 2; i++) { sd.stackPc12[i] = -1; sd.stackTo[i] = -1; }

    Move bestMove = MOVE_NONE, prevBest = MOVE_NONE;
    int prevScore = 0, stability = 0;
    int scoreChange = 0;
    int startDepth = 1 + (sd.threadId & 1); // helpers on odd ids start one deeper

    for (int depth = startDepth; depth <= searchLimits.depthLimit; depth++) {
        sd.selDepth = 0;
        sd.rootDepth = depth;
        int alpha = -INF, beta = INF, delta = ev(25);
        if (depth >= 4) { alpha = prevScore - delta; beta = prevScore + delta; }

        int score;
        while (true) {
            score = search(sd, pos, alpha, beta, depth, 0, true);
            if (Stopped) break;
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
                alpha = std::max(-INF, score - delta);
            } else if (score >= beta) {
                beta = std::min(INF, score + delta);
            } else break;
            delta += delta / 2;
        }
        if (Stopped) break;

        scoreChange = score - prevScore;
        prevScore = score;
        sd.lastScore = score;
        sd.completedDepth = depth;
        if (sd.pvLen[0] > 0) {
            bestMove = sd.pvTable[0][0];
            // A later iteration may be interrupted after search() has reset
            // pvLen[0]. Preserve this move separately so time expiration never
            // falls back to the first pseudo-legal move.
            sd.lastBest = bestMove;
        }

        if (!sd.isMain) continue; // helpers don't print or manage the clock

        if (bestMove == prevBest) stability = std::min(stability + 1, 8);
        else stability = 0;
        prevBest = bestMove;

        s64 ms = sd.elapsed();
        if (!SilentSearch) {
            u64 nodes = totalNodes();
            u64 nps = ms > 0 ? nodes * 1000 / ms : 0;
            std::cout << "info depth " << depth
                      << " seldepth " << sd.selDepth
                      << " score " << scoreString(score)
                      << " nodes " << nodes
                      << " nps " << nps
                      << " time " << ms
                      << " pv";
            for (int i = 0; i < sd.pvLen[0]; i++) std::cout << " " << moveStr(sd.pvTable[0][i]);
            std::cout << std::endl;
        }

        if (searchLimits.softLimit >= 0) {
            // Spend longer when a new iteration reveals unexpected danger;
            // save a little time when the position improves. The hard clock
            // limit remains unchanged, preserving the game reserve.
            double scorePressure = std::clamp(-scoreChange / double(ev(100)), -0.15, 0.50);
            double factor = 1.35 - 0.07 * stability + scorePressure;
            if (ms >= s64(searchLimits.softLimit * factor)) break;
        }
        if (std::abs(score) > MATE_BOUND && depth >= (MATE - std::abs(score)) + 10) break;
    }

    if (sd.isMain) Stopped = true; // main is done -> tell helpers to wrap up
}

// Drive a full search across the thread pool and return the best move.
// Runs the main thread in the caller; spawns Threads-1 helper threads.
inline Move searchPosition(Position& rootPos) {
    // Creating and joining helpers can dominate very short clock budgets and
    // even cause a flag under CPU contention.  Use SMP only when a move has
    // enough wall time for that startup cost to amortize.
    int n = (searchLimits.hardLimit >= 0 && searchLimits.hardLimit < 50)
          ? 1 : std::max(1, Threads);
    initThreadData(n);
    for (int i = 0; i < n; i++) {
        SearchData& td = *threadData[i];
        memcpy(td.keyHist, gameHist, gameHistLen * sizeof(u64));
        td.keyHistLen = gameHistLen;
    }

    std::vector<std::thread> helpers;
    for (int i = 1; i < n; i++)
        helpers.emplace_back([i, &rootPos] { iterativeDeepening(*threadData[i], rootPos); });

    iterativeDeepening(*threadData[0], rootPos); // main runs here; sets Stopped when done
    for (auto& h : helpers) h.join();

    // Helpers deliberately search different trees to seed the shared TT. Their
    // nominal depth is not comparable to the main iteration: root order and
    // aspiration failures can make a deeper helper result less reliable. The
    // deterministic main thread therefore remains the sole move authority.
    Move best = threadData[0]->lastBest;
    if (best == MOVE_NONE) {
        MoveList ml;
        genMoves<false>(rootPos, ml);
        Undo u;
        for (int i = 0; i < ml.count; i++)
            if (rootPos.make(ml.list[i].move, u)) { rootPos.unmake(ml.list[i].move, u); best = ml.list[i].move; break; }
    }
    return best;
}

// Single-threaded synchronous search (datagen): always one thread regardless of
// the Threads setting. Leaves the score in threadData[0]->lastScore.
inline Move searchSync(Position& rootPos) {
    initThreadData(1);
    SearchData& td = *threadData[0];
    memcpy(td.keyHist, gameHist, gameHistLen * sizeof(u64));
    td.keyHistLen = gameHistLen;
    iterativeDeepening(td, rootPos); // searches a copy; rootPos unchanged
    Move best = td.lastBest;
    if (best == MOVE_NONE) {
        MoveList ml;
        genMoves<false>(rootPos, ml);
        Undo u;
        for (int i = 0; i < ml.count; i++)
            if (rootPos.make(ml.list[i].move, u)) { rootPos.unmake(ml.list[i].move, u); best = ml.list[i].move; break; }
    }
    return best;
}
