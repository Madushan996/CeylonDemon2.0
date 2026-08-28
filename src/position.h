// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake. GNU GPL v3+; see LICENSE.
// Built on the NARC Engine codebase by the same author (GPLv3).
//
// CeylonDemon — position, make/unmake, move generation, SEE, perft
#pragma once
#include "resonance_acc.h"
#include "types.h"

// castling right bits
enum { WK_CASTLE = 1, WQ_CASTLE = 2, BK_CASTLE = 4, BQ_CASTLE = 8 };

// which rights survive when a move touches a square
inline int castlePerm[64];
inline void initCastlePerm() {
    for (int i = 0; i < 64; i++) castlePerm[i] = 15;
    castlePerm[0]  = 15 & ~WQ_CASTLE;             // a1
    castlePerm[4]  = 15 & ~(WK_CASTLE | WQ_CASTLE); // e1
    castlePerm[7]  = 15 & ~WK_CASTLE;             // h1
    castlePerm[56] = 15 & ~BQ_CASTLE;             // a8
    castlePerm[60] = 15 & ~(BK_CASTLE | BQ_CASTLE); // e8
    castlePerm[63] = 15 & ~BK_CASTLE;             // h8
}

struct Undo {
    u64 key;
    int castling;
    int epSquare;
    int halfmove;
    u8  captured;

    // Preserve the complete accumulator so unmake is one contiguous restore
    // instead of recalculating and reversing every feature delta.
    resonance::Accumulator savedAccumulator;
};

struct ScoredMove { Move move; int score; };

struct MoveList {
    ScoredMove list[256];
    int count = 0;
    void add(Move m) { list[count++].move = m; }
};

struct Position {
    u64 byColor[2];
    u64 byPiece[6];
    u8  board[64];      // piece type on square, NO_PIECE if empty
    int stm;            // side to move
    int epSquare;       // -1 if none
    int castling;
    int halfmove;
    int fullmove;
    u64 key;
    resonance::Accumulator acc; // four slices: AEGIS/LANCE x white/black

    void refreshAcc() {
        resonance::refreshAll(acc, byColor, byPiece, kingSq(WHITE), kingSq(BLACK));
    }

    // A move's effect on the accumulator, recorded during the board update and
    // applied afterwards. Applying it afterwards is what lets the incremental
    // path use the post-move king squares unconditionally: for a non-king move
    // the anchors have not moved, and for a king move that stayed inside its
    // bucket every index is unchanged anyway.
    //
    // Four is the worst case: castling moves a king and a rook.
    struct AccDelta { int c, pt, sq; bool add; };

    void applyAccDeltas(const AccDelta* d, int n, int skipMask) {
        const int wk = kingSq(WHITE), bk = kingSq(BLACK);
        for (int i = 0; i < n; i++) {
            if (d[i].add) resonance::addPiece(acc, d[i].c, d[i].pt, d[i].sq, wk, bk, skipMask);
            else          resonance::subPiece(acc, d[i].c, d[i].pt, d[i].sq, wk, bk, skipMask);
        }
    }

    // make() rebuilds the slices whose anchor king crossed a boundary.
    // unmake() does not: it restores them from the Undo record instead.
    void refreshMaskedSlices(int mask) {
        if (!mask) return;
        const int wk = kingSq(WHITE), bk = kingSq(BLACK);
        for (int h = 0; h < resonance::FRAME_NB; h++)
            for (int p = 0; p < 2; p++)
                if (mask & (1 << resonance::sliceBit(h, p))) {
                    const int anchor = resonance::anchorColor(h, p);
                    resonance::refreshSlice(acc, h, p, byColor, byPiece,
                                            anchor == WHITE ? wk : bk);
                }
    }

    u64 occupied() const { return byColor[WHITE] | byColor[BLACK]; }
    u64 pieces(int c, int pt) const { return byColor[c] & byPiece[pt]; }
    int kingSq(int c) const { return lsb(pieces(c, KING)); }
    int colorOn(int sq) const { return (byColor[WHITE] & bit(sq)) ? WHITE : BLACK; }

    void putPiece(int c, int pt, int sq) {
        byColor[c] |= bit(sq);
        byPiece[pt] |= bit(sq);
        board[sq] = u8(pt);
        key ^= Zpsq[c][pt][sq];
    }
    void removePiece(int c, int pt, int sq) {
        byColor[c] ^= bit(sq);
        byPiece[pt] ^= bit(sq);
        board[sq] = NO_PIECE;
        key ^= Zpsq[c][pt][sq];
    }
    void movePiece(int c, int pt, int from, int to) {
        u64 ft = bit(from) | bit(to);
        byColor[c] ^= ft;
        byPiece[pt] ^= ft;
        board[from] = NO_PIECE;
        board[to] = u8(pt);
        key ^= Zpsq[c][pt][from] ^ Zpsq[c][pt][to];
    }

    u64 attackersTo(int sq, u64 occ) const {
        return (pawnAtt[BLACK][sq] & pieces(WHITE, PAWN))
             | (pawnAtt[WHITE][sq] & pieces(BLACK, PAWN))
             | (knightAtt[sq] & byPiece[KNIGHT])
             | (kingAtt[sq] & byPiece[KING])
             | (bishopAttacks(sq, occ) & (byPiece[BISHOP] | byPiece[QUEEN]))
             | (rookAttacks(sq, occ) & (byPiece[ROOK] | byPiece[QUEEN]));
    }
    bool attacked(int sq, int byColorC) const {
        return attackersTo(sq, occupied()) & byColor[byColorC];
    }
    bool inCheck() const { return attacked(kingSq(stm), stm ^ 1); }

    bool hasNonPawnMaterial() const {
        return byColor[stm] & (byPiece[KNIGHT] | byPiece[BISHOP] | byPiece[ROOK] | byPiece[QUEEN]);
    }

    void clear() {
        memset(this, 0, sizeof(Position));
        for (int i = 0; i < 64; i++) board[i] = NO_PIECE;
        epSquare = -1;
        fullmove = 1;
    }

    void setFen(const std::string& fen) {
        clear();
        std::istringstream ss(fen);
        std::string boardStr, stmStr, castleStr, epStr;
        ss >> boardStr >> stmStr >> castleStr >> epStr;
        if (!(ss >> halfmove)) halfmove = 0;
        if (!(ss >> fullmove)) fullmove = 1;

        int sq = 56; // a8
        for (char ch : boardStr) {
            if (ch == '/') { sq -= 16; continue; }
            if (ch >= '1' && ch <= '8') { sq += ch - '0'; continue; }
            int c = isupper(ch) ? WHITE : BLACK;
            int pt;
            switch (tolower(ch)) {
                case 'p': pt = PAWN; break;   case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break; case 'r': pt = ROOK; break;
                case 'q': pt = QUEEN; break;  default:  pt = KING; break;
            }
            putPiece(c, pt, sq++);
        }
        stm = (stmStr == "w") ? WHITE : BLACK;
        if (stm == BLACK) key ^= Zside;
        castling = 0;
        for (char ch : castleStr) {
            if (ch == 'K') castling |= WK_CASTLE;
            else if (ch == 'Q') castling |= WQ_CASTLE;
            else if (ch == 'k') castling |= BK_CASTLE;
            else if (ch == 'q') castling |= BQ_CASTLE;
        }
        key ^= Zcastle[castling];
        epSquare = -1;
        if (epStr.size() == 2) {
            epSquare = (epStr[0] - 'a') + 8 * (epStr[1] - '1');
            key ^= ZepFile[fileOf(epSquare)];
        }
        refreshAcc();
    }

    // returns false (and restores nothing-changed state) if move leaves own king in check
    bool make(Move m, Undo& u) {
        u.key = key; u.castling = castling; u.epSquare = epSquare; u.halfmove = halfmove;
        u.captured = NO_PIECE;
        int from = fromSq(m), to = toSq(m), flag = flagOf(m);
        int us = stm, them = us ^ 1;
        int pt = board[from];

        key ^= Zside;
        if (epSquare != -1) { key ^= ZepFile[fileOf(epSquare)]; epSquare = -1; }
        halfmove++;

        bool nn = resonance::weightsLoaded;
        if (nn) memcpy(&u.savedAccumulator, &acc, sizeof(acc));

        // Accumulator deltas are collected here and applied once the board is
        // fully updated, so they always see the post-move king squares.
        AccDelta d[4];
        int nd = 0;
        int kingFrom = -1, kingTo = -1;

        if (flag == CASTLING) {
            // to is king destination: g1/c1/g8/c8
            movePiece(us, KING, from, to);
            int rf = 0, rt = 0;
            switch (to) {
                case 6:  rf = 7;  rt = 5;  break; // g1
                case 2:  rf = 0;  rt = 3;  break; // c1
                case 62: rf = 63; rt = 61; break; // g8
                case 58: rf = 56; rt = 59; break; // c8
            }
            movePiece(us, ROOK, rf, rt);
            if (nn) {
                d[nd++] = {us, KING, from, false}; d[nd++] = {us, KING, to, true};
                d[nd++] = {us, ROOK, rf, false};   d[nd++] = {us, ROOK, rt, true};
            }
            kingFrom = from; kingTo = to;
        } else {
            if (flag == ENPASSANT) {
                int capSq = to + (us == WHITE ? -8 : 8);
                removePiece(them, PAWN, capSq);
                if (nn) d[nd++] = {them, PAWN, capSq, false};
                u.captured = NO_PIECE; // captured pawn implied by flag
                halfmove = 0;
            } else if (board[to] != NO_PIECE) {
                u.captured = board[to];
                removePiece(them, board[to], to);
                if (nn) d[nd++] = {them, u.captured, to, false};
                halfmove = 0;
            }
            movePiece(us, pt, from, to);
            int placed = pt;
            if (pt == PAWN) {
                halfmove = 0;
                if ((to ^ from) == 16) { // double push
                    epSquare = (from + to) / 2;
                    key ^= ZepFile[fileOf(epSquare)];
                } else if (flag == PROMOTION) {
                    removePiece(us, PAWN, to);
                    putPiece(us, promoOf(m), to);
                    placed = promoOf(m);
                }
            }
            if (nn) {
                d[nd++] = {us, pt, from, false};
                d[nd++] = {us, placed, to, true};
            }
            if (pt == KING) { kingFrom = from; kingTo = to; }
        }

        if (nn) {
            const int mask =
                kingFrom >= 0 ? resonance::refreshMaskForKingMove(us, kingFrom, kingTo) : 0;
            applyAccDeltas(d, nd, mask);
            refreshMaskedSlices(mask);
        }

        key ^= Zcastle[castling];
        castling &= castlePerm[from] & castlePerm[to];
        key ^= Zcastle[castling];

        stm = them;
        if (us == BLACK) fullmove++;

        if (attacked(kingSq(us), them)) { unmake(m, u); return false; }
        return true;
    }

    void unmake(Move m, const Undo& u) {
        int from = fromSq(m), to = toSq(m), flag = flagOf(m);
        int them = stm, us = stm ^ 1; // us = side that moved
        bool nn = resonance::weightsLoaded;

        if (flag == CASTLING) {
            movePiece(us, KING, to, from);
            int rf = 0, rt = 0;
            switch (to) {
                case 6:  rf = 7;  rt = 5;  break;
                case 2:  rf = 0;  rt = 3;  break;
                case 62: rf = 63; rt = 61; break;
                case 58: rf = 56; rt = 59; break;
            }
            movePiece(us, ROOK, rt, rf);
        } else {
            if (flag == PROMOTION) {
                removePiece(us, promoOf(m), to);
                putPiece(us, PAWN, to);
            }
            movePiece(us, board[to], to, from);
            if (flag == ENPASSANT) {
                int capSq = to + (us == WHITE ? -8 : 8);
                putPiece(them, PAWN, capSq);
            } else if (u.captured != NO_PIECE) {
                putPiece(them, u.captured, to);
            }
        }

        stm = us;
        if (us == BLACK) fullmove--;
        key = u.key; castling = u.castling; epSquare = u.epSquare; halfmove = u.halfmove;

        if (nn) memcpy(&acc, &u.savedAccumulator, sizeof(acc));
    }

    void makeNull(Undo& u) {
        u.key = key; u.castling = castling; u.epSquare = epSquare; u.halfmove = halfmove;
        u.captured = NO_PIECE;
        key ^= Zside;
        if (epSquare != -1) { key ^= ZepFile[fileOf(epSquare)]; epSquare = -1; }
        halfmove++;
        stm ^= 1;
    }
    void unmakeNull(const Undo& u) {
        stm ^= 1;
        key = u.key; castling = u.castling; epSquare = u.epSquare; halfmove = u.halfmove;
    }

    bool isCapture(Move m) const {
        return board[toSq(m)] != NO_PIECE || flagOf(m) == ENPASSANT;
    }

    std::string fen() const {
        std::ostringstream os;
        for (int r = 7; r >= 0; r--) {
            int empty = 0;
            for (int f = 0; f < 8; f++) {
                int sq = r * 8 + f;
                if (board[sq] == NO_PIECE) { empty++; continue; }
                if (empty) { os << empty; empty = 0; }
                char c = "pnbrqk"[board[sq]];
                os << char(colorOn(sq) == WHITE ? toupper(c) : c);
            }
            if (empty) os << empty;
            if (r) os << '/';
        }
        os << (stm == WHITE ? " w " : " b ");
        if (!castling) os << '-';
        else {
            if (castling & WK_CASTLE) os << 'K';
            if (castling & WQ_CASTLE) os << 'Q';
            if (castling & BK_CASTLE) os << 'k';
            if (castling & BQ_CASTLE) os << 'q';
        }
        os << ' ' << (epSquare == -1 ? "-" : sqStr(epSquare));
        os << ' ' << halfmove << ' ' << fullmove;
        return os.str();
    }
};

// ----------------------------- Move generation ------------------------------
template<bool CapturesOnly>
inline void genMoves(const Position& pos, MoveList& ml) {
    int us = pos.stm, them = us ^ 1;
    u64 occ = pos.occupied();
    u64 ours = pos.byColor[us];
    u64 theirs = pos.byColor[them];
    u64 targets = CapturesOnly ? theirs : ~ours;

    // pawns
    u64 pawns = pos.pieces(us, PAWN);
    int up = (us == WHITE) ? 8 : -8;
    u64 promoRank = (us == WHITE) ? RANK_8 : RANK_1;
    u64 thirdRank = (us == WHITE) ? RANK_3 : RANK_6;

    u64 single = (us == WHITE ? pawns << 8 : pawns >> 8) & ~occ;
    if (!CapturesOnly) {
        u64 dbl = (us == WHITE ? (single & thirdRank) << 8 : (single & thirdRank) >> 8) & ~occ;
        u64 quiet = single & ~promoRank;
        while (quiet) { int to = poplsb(quiet); ml.add(makeMove(to - up, to)); }
        while (dbl) { int to = poplsb(dbl); ml.add(makeMove(to - 2 * up, to)); }
    }
    // promotions by push (generated even in captures-only - they're tactical)
    u64 promoPush = single & promoRank;
    while (promoPush) {
        int to = poplsb(promoPush);
        for (int p = QUEEN; p >= KNIGHT; p--) ml.add(makeMove(to - up, to, PROMOTION, p - KNIGHT));
    }
    // pawn captures
    u64 capL = (us == WHITE ? (pawns & ~FILE_A) << 7 : (pawns & ~FILE_A) >> 9) & theirs;
    u64 capR = (us == WHITE ? (pawns & ~FILE_H) << 9 : (pawns & ~FILE_H) >> 7) & theirs;
    int dl = (us == WHITE) ? 7 : -9;
    int dr = (us == WHITE) ? 9 : -7;
    u64 c;
    c = capL & ~promoRank; while (c) { int to = poplsb(c); ml.add(makeMove(to - dl, to)); }
    c = capR & ~promoRank; while (c) { int to = poplsb(c); ml.add(makeMove(to - dr, to)); }
    c = capL & promoRank;
    while (c) { int to = poplsb(c); for (int p = QUEEN; p >= KNIGHT; p--) ml.add(makeMove(to - dl, to, PROMOTION, p - KNIGHT)); }
    c = capR & promoRank;
    while (c) { int to = poplsb(c); for (int p = QUEEN; p >= KNIGHT; p--) ml.add(makeMove(to - dr, to, PROMOTION, p - KNIGHT)); }
    // en passant
    if (pos.epSquare != -1) {
        u64 epAttackers = pawnAtt[them][pos.epSquare] & pawns;
        while (epAttackers) { int from = poplsb(epAttackers); ml.add(makeMove(from, pos.epSquare, ENPASSANT)); }
    }

    // knights
    u64 b = pos.pieces(us, KNIGHT);
    while (b) {
        int from = poplsb(b);
        u64 att = knightAtt[from] & targets;
        while (att) ml.add(makeMove(from, poplsb(att)));
    }
    // bishops
    b = pos.pieces(us, BISHOP);
    while (b) {
        int from = poplsb(b);
        u64 att = bishopAttacks(from, occ) & targets;
        while (att) ml.add(makeMove(from, poplsb(att)));
    }
    // rooks
    b = pos.pieces(us, ROOK);
    while (b) {
        int from = poplsb(b);
        u64 att = rookAttacks(from, occ) & targets;
        while (att) ml.add(makeMove(from, poplsb(att)));
    }
    // queens
    b = pos.pieces(us, QUEEN);
    while (b) {
        int from = poplsb(b);
        u64 att = queenAttacks(from, occ) & targets;
        while (att) ml.add(makeMove(from, poplsb(att)));
    }
    // king
    int ksq = pos.kingSq(us);
    u64 att = kingAtt[ksq] & targets;
    while (att) ml.add(makeMove(ksq, poplsb(att)));

    // castling
    if (!CapturesOnly) {
        if (us == WHITE) {
            if ((pos.castling & WK_CASTLE) && !(occ & 0x60ULL)
                && !pos.attacked(4, BLACK) && !pos.attacked(5, BLACK))
                ml.add(makeMove(4, 6, CASTLING));
            if ((pos.castling & WQ_CASTLE) && !(occ & 0x0EULL)
                && !pos.attacked(4, BLACK) && !pos.attacked(3, BLACK))
                ml.add(makeMove(4, 2, CASTLING));
        } else {
            if ((pos.castling & BK_CASTLE) && !(occ & 0x6000000000000000ULL)
                && !pos.attacked(60, WHITE) && !pos.attacked(61, WHITE))
                ml.add(makeMove(60, 62, CASTLING));
            if ((pos.castling & BQ_CASTLE) && !(occ & 0x0E00000000000000ULL)
                && !pos.attacked(60, WHITE) && !pos.attacked(59, WHITE))
                ml.add(makeMove(60, 58, CASTLING));
        }
    }
}

// ----------------------------- SEE ------------------------------------------
constexpr int seeValue[7] = {100, 320, 330, 500, 900, 20000, 0};

inline int see(const Position& pos, Move m) {
    int from = fromSq(m), to = toSq(m), flag = flagOf(m);
    if (flag == CASTLING) return 0;

    int gain[32];
    int d = 0;
    u64 occ = pos.occupied();
    int side = pos.stm;

    int capturedPt = (flag == ENPASSANT) ? PAWN : pos.board[to];
    gain[0] = (capturedPt == NO_PIECE) ? 0 : seeValue[capturedPt];
    int curPt = pos.board[from];
    if (flag == PROMOTION) {
        curPt = promoOf(m);
        gain[0] += seeValue[curPt] - seeValue[PAWN];
    }

    if (flag == ENPASSANT) occ ^= bit(to + (side == WHITE ? -8 : 8));
    occ ^= bit(from);

    u64 attackers = pos.attackersTo(to, occ) & occ;
    side ^= 1;

    while (true) {
        u64 sideAtt = attackers & pos.byColor[side];
        if (!sideAtt) break;
        int pt;
        u64 bb = 0;
        for (pt = PAWN; pt <= KING; pt++) {
            bb = sideAtt & pos.byPiece[pt];
            if (bb) break;
        }
        d++;
        gain[d] = seeValue[curPt] - gain[d - 1];
        curPt = pt;
        occ ^= bb & -bb;
        // discovered (x-ray) attackers
        if (pt == PAWN || pt == BISHOP || pt == QUEEN)
            attackers |= bishopAttacks(to, occ) & (pos.byPiece[BISHOP] | pos.byPiece[QUEEN]);
        if (pt == ROOK || pt == QUEEN)
            attackers |= rookAttacks(to, occ) & (pos.byPiece[ROOK] | pos.byPiece[QUEEN]);
        attackers &= occ;
        side ^= 1;
        if (d >= 30) break;
    }
    // Fold every recapture back to the initial move.  The previous pre-
    // decrement loop stopped at d == 1, so a single defender was ignored
    // entirely (for example QxP defended by a pawn looked profitable).
    while (d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        --d;
    }
    return gain[0];
}

// ----------------------------- Perft ----------------------------------------
inline u64 perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    genMoves<false>(pos, ml);
    u64 nodes = 0;
    Undo u;
    for (int i = 0; i < ml.count; i++) {
        if (!pos.make(ml.list[i].move, u)) continue;
        nodes += (depth == 1) ? 1 : perft(pos, depth - 1);
        pos.unmake(ml.list[i].move, u);
    }
    return nodes;
}
