// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Built on the NARC Engine codebase by the same author (GPLv3).
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// CeylonDemon — basic types, bitboards, attack tables, zobrist
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;
using s64 = int64_t;

enum Color { WHITE, BLACK };
enum PieceType { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE };

constexpr int MAX_PLY = 128;
constexpr int MATE    = 30000;
constexpr int MATE_BOUND = 29000; // scores beyond this are mate scores
constexpr int INF     = 31000;
constexpr int VALUE_DRAW = 0;

constexpr u64 FILE_A = 0x0101010101010101ULL;
constexpr u64 FILE_H = 0x8080808080808080ULL;
constexpr u64 RANK_1 = 0x00000000000000FFULL;
constexpr u64 RANK_2 = 0x000000000000FF00ULL;
constexpr u64 RANK_3 = 0x0000000000FF0000ULL;
constexpr u64 RANK_6 = 0x0000FF0000000000ULL;
constexpr u64 RANK_7 = 0x00FF000000000000ULL;
constexpr u64 RANK_8 = 0xFF00000000000000ULL;

inline int popcount(u64 b) { return __builtin_popcountll(b); }
inline int lsb(u64 b)      { return __builtin_ctzll(b); }
inline int poplsb(u64& b)  { int s = lsb(b); b &= b - 1; return s; }
inline u64 bit(int sq)     { return 1ULL << sq; }
inline int fileOf(int sq)  { return sq & 7; }
inline int rankOf(int sq)  { return sq >> 3; }

// ----------------------------- Moves ---------------------------------------
// 16-bit move: bits 0-5 from, 6-11 to, 12-13 promo piece (0=N..3=Q), 14-15 flag
enum MoveFlag { NORMAL = 0, PROMOTION = 1, ENPASSANT = 2, CASTLING = 3 };
using Move = u16;
constexpr Move MOVE_NONE = 0;

inline Move makeMove(int from, int to, int flag = NORMAL, int promo = 0) {
    return Move(from | (to << 6) | (promo << 12) | (flag << 14));
}
inline int fromSq(Move m)  { return m & 63; }
inline int toSq(Move m)    { return (m >> 6) & 63; }
inline int promoOf(Move m) { return ((m >> 12) & 3) + KNIGHT; }
inline int flagOf(Move m)  { return (m >> 14) & 3; }

inline std::string sqStr(int sq) {
    return { char('a' + fileOf(sq)), char('1' + rankOf(sq)) };
}
inline std::string moveStr(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s = sqStr(fromSq(m)) + sqStr(toSq(m));
    if (flagOf(m) == PROMOTION) s += "nbrq"[promoOf(m) - KNIGHT];
    return s;
}

// ----------------------------- PRNG ----------------------------------------
struct PRNG {
    u64 s;
    explicit PRNG(u64 seed) : s(seed) {}
    u64 rand64() { // xorshift*
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    u64 sparse() { return rand64() & rand64() & rand64(); }
};

// ----------------------------- Attack tables --------------------------------
inline u64 pawnAtt[2][64];
inline u64 knightAtt[64];
inline u64 kingAtt[64];

struct Magic {
    u64  mask;
    u64  magic;
    int  shift;
    u64* attacks;
    int index(u64 occ) const { return int(((occ & mask) * magic) >> shift); }
};
inline Magic rookMagics[64];
inline Magic bishopMagics[64];
inline u64 rookTable[0x19000];   // 102400
inline u64 bishopTable[0x1480];  // 5248

inline u64 slidingAttack(int sq, u64 occ, bool rook) {
    static const int dirs[2][4][2] = {
        {{1,1},{1,-1},{-1,1},{-1,-1}},   // bishop
        {{1,0},{-1,0},{0,1},{0,-1}}      // rook
    };
    u64 att = 0;
    for (int d = 0; d < 4; d++) {
        int r = rankOf(sq) + dirs[rook][d][0];
        int f = fileOf(sq) + dirs[rook][d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            int s = r * 8 + f;
            att |= bit(s);
            if (occ & bit(s)) break;
            r += dirs[rook][d][0];
            f += dirs[rook][d][1];
        }
    }
    return att;
}

inline u64 relevantMask(int sq, bool rook) {
    u64 m = 0;
    int rk = rankOf(sq), fl = fileOf(sq);
    if (rook) {
        for (int r = rk + 1; r <= 6; r++) m |= bit(r * 8 + fl);
        for (int r = rk - 1; r >= 1; r--) m |= bit(r * 8 + fl);
        for (int f = fl + 1; f <= 6; f++) m |= bit(rk * 8 + f);
        for (int f = fl - 1; f >= 1; f--) m |= bit(rk * 8 + f);
    } else {
        for (int r = rk + 1, f = fl + 1; r <= 6 && f <= 6; r++, f++) m |= bit(r * 8 + f);
        for (int r = rk + 1, f = fl - 1; r <= 6 && f >= 1; r++, f--) m |= bit(r * 8 + f);
        for (int r = rk - 1, f = fl + 1; r >= 1 && f <= 6; r--, f++) m |= bit(r * 8 + f);
        for (int r = rk - 1, f = fl - 1; r >= 1 && f >= 1; r--, f--) m |= bit(r * 8 + f);
    }
    return m;
}

inline void initMagics(bool rook) {
    PRNG rng(0x6C078965ULL);
    u64* table = rook ? rookTable : bishopTable;
    int offset = 0;
    for (int sq = 0; sq < 64; sq++) {
        Magic& mg = rook ? rookMagics[sq] : bishopMagics[sq];
        mg.mask  = relevantMask(sq, rook);
        int bits = popcount(mg.mask);
        mg.shift = 64 - bits;
        mg.attacks = table + offset;
        int size = 1 << bits;

        u64 occs[4096], refs[4096];
        u64 b = 0;
        int n = 0;
        do {
            occs[n] = b;
            refs[n] = slidingAttack(sq, b, rook);
            n++;
            b = (b - mg.mask) & mg.mask;
        } while (b);

        // search a working magic
        std::vector<int> epoch(size, -1);
        int tries = 0;
        while (true) {
            mg.magic = rng.sparse();
            if (popcount((mg.mask * mg.magic) >> 56) < 6) continue;
            bool ok = true;
            for (int i = 0; i < n; i++) {
                int idx = mg.index(occs[i]);
                if (epoch[idx] != tries) { epoch[idx] = tries; mg.attacks[idx] = refs[i]; }
                else if (mg.attacks[idx] != refs[i]) { ok = false; break; }
            }
            if (ok) break;
            tries++;
        }
        offset += size;
    }
}

inline u64 bishopAttacks(int sq, u64 occ) { const Magic& m = bishopMagics[sq]; return m.attacks[m.index(occ)]; }
inline u64 rookAttacks(int sq, u64 occ)   { const Magic& m = rookMagics[sq];   return m.attacks[m.index(occ)]; }
inline u64 queenAttacks(int sq, u64 occ)  { return bishopAttacks(sq, occ) | rookAttacks(sq, occ); }

// masks for eval
inline u64 fileMask[8];
inline u64 adjacentFilesMask[8];
inline u64 passedMask[2][64]; // squares that must be free of enemy pawns

inline void initTables() {
    for (int sq = 0; sq < 64; sq++) {
        u64 b = bit(sq);
        pawnAtt[WHITE][sq] = ((b & ~FILE_A) << 7) | ((b & ~FILE_H) << 9);
        pawnAtt[BLACK][sq] = ((b & ~FILE_A) >> 9) | ((b & ~FILE_H) >> 7);

        u64 n = 0;
        const int nd[8][2] = {{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
        for (auto& d : nd) {
            int r = rankOf(sq) + d[0], f = fileOf(sq) + d[1];
            if (r >= 0 && r < 8 && f >= 0 && f < 8) n |= bit(r * 8 + f);
        }
        knightAtt[sq] = n;

        u64 k = 0;
        for (int dr = -1; dr <= 1; dr++)
            for (int df = -1; df <= 1; df++) {
                if (!dr && !df) continue;
                int r = rankOf(sq) + dr, f = fileOf(sq) + df;
                if (r >= 0 && r < 8 && f >= 0 && f < 8) k |= bit(r * 8 + f);
            }
        kingAtt[sq] = k;
    }
    for (int f = 0; f < 8; f++) {
        fileMask[f] = FILE_A << f;
        adjacentFilesMask[f] = (f > 0 ? FILE_A << (f - 1) : 0) | (f < 7 ? FILE_A << (f + 1) : 0);
    }
    for (int sq = 0; sq < 64; sq++) {
        u64 span = fileMask[fileOf(sq)] | adjacentFilesMask[fileOf(sq)];
        u64 ahead = rankOf(sq) < 7 ? (~0ULL << (8 * (rankOf(sq) + 1))) : 0;
        u64 behind = rankOf(sq) ? (~0ULL >> (8 * (8 - rankOf(sq)))) : 0;
        passedMask[WHITE][sq] = span & ahead;
        passedMask[BLACK][sq] = span & behind;
    }
    initMagics(false);
    initMagics(true);
}

// ----------------------------- Zobrist --------------------------------------
inline u64 Zpsq[2][6][64];
inline u64 Zcastle[16];
inline u64 ZepFile[8];
inline u64 Zside;

inline void initZobrist() {
    PRNG rng(0x9E3779B97F4A7C15ULL);
    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 6; p++)
            for (int s = 0; s < 64; s++) Zpsq[c][p][s] = rng.rand64();
    for (int i = 0; i < 16; i++) Zcastle[i] = rng.rand64();
    for (int i = 0; i < 8; i++)  ZepFile[i] = rng.rand64();
    Zside = rng.rand64();
}
