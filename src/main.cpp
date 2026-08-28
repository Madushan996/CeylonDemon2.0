// SPDX-License-Identifier: GPL-3.0-or-later
//
// CeylonDemon — a UCI chess engine
// Copyright (C) 2026 Madushan Dissanayake
// Built on the NARC Engine codebase by the same author (GPLv3).
// Attribution and lineage: see NOTICE.md.
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. This program is distributed WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License (LICENSE file) for more details.
//
// Build: see build.ps1
#include "search.h"
#include <fstream>
#include <random>
#include <thread>

static Position rootPos;
static std::thread searchThread;
static int moveOverhead = 30; // ms

static void stopSearchThread() {
    Stopped = true;
    if (searchThread.joinable()) searchThread.join();
}

static Move parseUciMove(const Position& pos, const std::string& str) {
    MoveList ml;
    genMoves<false>(pos, ml);
    for (int i = 0; i < ml.count; i++)
        if (moveStr(ml.list[i].move) == str) return ml.list[i].move;
    return MOVE_NONE;
}

static void setPosition(std::istringstream& is) {
    std::string token;
    is >> token;
    std::string fen;
    if (token == "startpos") {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        is >> token; // consume "moves" if present
    } else if (token == "fen") {
        while (is >> token && token != "moves") fen += token + " ";
    }
    rootPos.setFen(fen);
    gameHistLen = 0;
    gameHist[gameHistLen++] = rootPos.key;

    Undo u;
    while (is >> token) {
        Move m = parseUciMove(rootPos, token);
        if (m == MOVE_NONE) break;
        if (!rootPos.make(m, u)) break; // illegal move in the list: stop here
        gameHist[gameHistLen++] = rootPos.key;
        if (gameHistLen >= 1000) { // keep buffer safe; old entries no longer needed
            memmove(gameHist, gameHist + 500, 500 * sizeof(u64));
            gameHistLen -= 500;
        }
    }
}

static void startSearch(const SearchLimits& limits) {
    stopSearchThread();
    Stopped = false;
    searchLimits = limits;
    searchStart = std::chrono::steady_clock::now();
    searchThread = std::thread([] {
        Move best = searchPosition(rootPos); // Lazy SMP across `Threads` threads
        // in infinite mode, UCI requires waiting for "stop" before bestmove
        while (searchLimits.infinite && !Stopped)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::cout << "bestmove " << moveStr(best) << std::endl;
    });
}

static void goCommand(std::istringstream& is) {
    SearchLimits limits;
    s64 wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;
    int movestogo = 0;
    std::string token;

    while (is >> token) {
        if (token == "wtime") is >> wtime;
        else if (token == "btime") is >> btime;
        else if (token == "winc") is >> winc;
        else if (token == "binc") is >> binc;
        else if (token == "movestogo") is >> movestogo;
        else if (token == "movetime") is >> movetime;
        else if (token == "depth") is >> limits.depthLimit;
        else if (token == "nodes") is >> limits.nodeLimit;
        else if (token == "infinite") limits.infinite = true;
        else if (token == "perft") {
            int d; is >> d;
            auto t0 = std::chrono::steady_clock::now();
            u64 n = perft(rootPos, d);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "perft " << d << ": " << n << " nodes, " << ms << " ms, "
                      << (ms > 0 ? n * 1000 / ms : 0) << " nps" << std::endl;
            return;
        }
    }

    s64 myTime = rootPos.stm == WHITE ? wtime : btime;
    s64 myInc  = rootPos.stm == WHITE ? winc : binc;

    if (movetime >= 0) {
        // UCI "movetime" is an explicit analysis budget, not a game clock.
        // Move Overhead applies only to wtime/btime where flag protection is
        // needed. Subtracting it here cost 30% of a 100 ms test budget.
        limits.softLimit = limits.hardLimit = std::max<s64>(1, movetime);
    } else if (myTime >= 0) {
        myTime = std::max<s64>(1, myTime - moveOverhead);
        // Without an explicit movestogo, retain a deeper reserve in the
        // opening and spend it progressively as the game advances.
        int estimated = std::clamp(50 - rootPos.fullmove / 2, 20, 40);
        int mtg = movestogo > 0 ? std::clamp(movestogo, 2, 40) : estimated;
        bool conversionTime = popcount(rootPos.occupied()) <= 10
                           && std::abs(evaluate(rootPos)) >= 400;
        // Increment is replenished after every completed move. Discounting it
        // to 75% left several seconds unused in ordinary 10s+100ms games and
        // effectively gave the opponent persistent time odds.
        s64 usableInc = conversionTime ? myInc * 7 / 8 : myInc;
        s64 base = myTime / mtg + usableInc;
        // The previous 50%-of-clock hard cap could spend a full second from a
        // two-second clock whenever one iteration ran long. Keep the normal
        // budget slightly less conservative, but cap exceptional moves at 20%
        // of the remaining clock so increment games cannot spiral into flags.
        limits.softLimit = std::max<s64>(1, base);
        // In increment games, the old 20%-of-clock cap became far too
        // conservative once the remaining clock approached the increment.
        // Normally spend up to 75% of the increment and retain 20% of the
        // clock. In an already decisive sparse ending, use up to 87.5% of the
        // increment while keeping a smaller 10% flag-safety reserve.
        s64 clockBudget = conversionTime ? myTime * 9 / 10 : myTime * 4 / 5;
        s64 reserveCap = conversionTime ? myTime / 10 : myTime / 5;
        s64 incrementBudget = std::min(clockBudget, usableInc);
        s64 safeHardCap = std::max(reserveCap, incrementBudget);
        limits.hardLimit = std::max<s64>(1, std::min(base * 3, safeHardCap));
        limits.softLimit = std::min(limits.softLimit, limits.hardLimit);
    }
    startSearch(limits);
}

// ----------------------------- Training data generation ----------------------
// datagen <games> <outfile> [nodes_per_move]
// Self-play with random openings at fixed nodes; writes "fen | score | result"
// (score in cp and result 1/0.5/0 both from White's point of view).
static void datagenCmd(std::istringstream& is) {
    int games = 1000;
    std::string outPath = "ceylondemon_data.txt";
    long long nodes = 5000;
    is >> games >> outPath >> nodes;

    std::ofstream out(outPath, std::ios::app);
    if (!out) { std::cout << "datagen: cannot open " << outPath << std::endl; return; }

    std::random_device rd;
    PRNG rng(u64(rd()) << 32 ^ u64(rd()) ^
             std::chrono::steady_clock::now().time_since_epoch().count());

    SilentSearch = true;
    tt.resize(16); // small hash: cleared per game
    u64 totalPos = 0;
    auto t0 = std::chrono::steady_clock::now();

    struct Sample { std::string fen; int wscore; };
    std::vector<Sample> samples;
    Undo u;

    for (int g = 1; g <= games; g++) {
        tt.clear();
        clearAllHistories();
        rootPos.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        gameHistLen = 0;
        gameHist[gameHistLen++] = rootPos.key;

        // random opening (8-10 plies)
        int openPlies = 8 + int(rng.rand64() % 3);
        bool ok = true;
        for (int i = 0; i < openPlies; i++) {
            MoveList ml;
            genMoves<false>(rootPos, ml);
            Move legal[256];
            int n = 0;
            for (int j = 0; j < ml.count; j++)
                if (rootPos.make(ml.list[j].move, u)) {
                    rootPos.unmake(ml.list[j].move, u);
                    legal[n++] = ml.list[j].move;
                }
            if (n == 0) { ok = false; break; }
            rootPos.make(legal[rng.rand64() % n], u);
            gameHist[gameHistLen++] = rootPos.key;
        }
        if (!ok) { g--; continue; }

        // skip badly unbalanced openings
        Stopped = false;
        searchLimits = SearchLimits{};
        searchLimits.nodeLimit = 1500;
        searchStart = std::chrono::steady_clock::now();
        searchSync(rootPos);
        if (std::abs(threadData[0]->lastScore) > 500) { g--; continue; }

        samples.clear();
        double result = 0.5;
        int adjCount = 0;

        for (int ply = 0; ply < 300; ply++) {
            // game over?
            MoveList ml;
            genMoves<false>(rootPos, ml);
            int n = 0;
            for (int j = 0; j < ml.count && !n; j++)
                if (rootPos.make(ml.list[j].move, u)) { rootPos.unmake(ml.list[j].move, u); n++; }
            if (n == 0) {
                result = rootPos.inCheck() ? (rootPos.stm == WHITE ? 0.0 : 1.0) : 0.5;
                break;
            }
            int reps = 0;
            for (int j = 0; j < gameHistLen; j++) reps += gameHist[j] == rootPos.key;
            if (reps >= 3 || rootPos.halfmove >= 100 || insufficientMaterial(rootPos))
                break; // draw

            Stopped = false;
            searchLimits = SearchLimits{};
            searchLimits.nodeLimit = nodes;
            searchLimits.depthLimit = 32;
            searchStart = std::chrono::steady_clock::now();
            Move best = searchSync(rootPos);
            int score = threadData[0]->lastScore; // side-to-move pov
            int wscore = rootPos.stm == WHITE ? score : -score;

            // win adjudication: decisive score held for 4 consecutive plies
            if (std::abs(score) >= 1500) {
                if (++adjCount >= 4) { result = wscore > 0 ? 1.0 : 0.0; break; }
            } else adjCount = 0;

            if (!rootPos.inCheck() && !rootPos.isCapture(best)
                && flagOf(best) != PROMOTION && std::abs(score) < 1500)
                samples.push_back({rootPos.fen(), wscore});

            rootPos.make(best, u);
            gameHist[gameHistLen++] = rootPos.key;
        }

        for (auto& s : samples)
            out << s.fen << " | " << s.wscore << " | " << result << "\n";
        totalPos += samples.size();

        if (g % 25 == 0) {
            out.flush();
            auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "datagen: " << g << "/" << games << " games, "
                      << totalPos << " positions, "
                      << (secs ? totalPos / secs : 0) << " pos/s" << std::endl;
        }
    }
    out.flush();
    SilentSearch = false;
    std::cout << "datagen done: " << totalPos << " positions -> " << outPath << std::endl;
}

static void uciLoop() {
    std::string line, token;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        token.clear();
        is >> std::skipws >> token;

        if (token == "uci") {
            std::cout << "id name CeylonDemon 2.0\n"
                      << "id author Madushan Dissanayake\n"
                      << "option name Hash type spin default 128 min 1 max 4096\n"
                      << "option name Threads type spin default 1 min 1 max 256\n"
                      << "option name Move Overhead type spin default 30 min 0 max 1000\n"
                      << "option name EvalFile type string default <embedded>\n";
            std::cout << "info string "
                      << (resonance::weightsLoaded
                              ? "Resonance v10 active (" + resonance::netInfo() + ")"
                              : std::string("no network loaded"))
                      << "\n";
            std::cout << "uciok" << std::endl;
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "setoption") {
            std::string name, value;
            is >> token; // "name"
            while (is >> token && token != "value") name += (name.empty() ? "" : " ") + token;
            while (is >> token) value += (value.empty() ? "" : " ") + token;
            if (name == "Hash") {
                stopSearchThread();
                tt.resize(std::clamp(std::stoi(value), 1, 4096));
            } else if (name == "Move Overhead") {
                moveOverhead = std::clamp(std::stoi(value), 0, 1000);
            } else if (name == "Threads") {
                stopSearchThread();
                Threads = std::clamp(std::stoi(value), 1, 256);
                initThreadData(Threads);
            } else if (name == "EvalFile") {
                // try the external file; if absent/invalid, keep the built-in net
                bool fromFile = !value.empty() && resonance::loadFromFile(value);
                if (!fromFile) resonance::loadEmbedded();
                rootPos.refreshAcc();
                if (fromFile)
                    std::cout << "info string network loaded from file: " << value << std::endl;
                else if (resonance::weightsLoaded)
                    std::cout << "info string Resonance v10 active ("
                              << resonance::netInfo() << ")" << std::endl;
                else
                    std::cout << "info string no network loaded" << std::endl;
            }
        } else if (token == "ucinewgame") {
            stopSearchThread();
            tt.clear();
            clearAllHistories();
        } else if (token == "position") {
            stopSearchThread();
            setPosition(is);
        } else if (token == "go") {
            goCommand(is);
        } else if (token == "stop") {
            Stopped = true;
        } else if (token == "quit") {
            break;
        } else if (token == "datagen") {
            datagenCmd(is);
        } else if (token == "eval") {
            int sigma = 0;
            int e1 = evaluate(rootPos, &sigma);
            std::cout << "Blended score (side to move): " << e1 << "\n"
                      << "Head disagreement (sigma):    " << sigma << std::endl;
            if (resonance::weightsLoaded) { // verify incremental accumulator against refresh
                rootPos.refreshAcc();
                int e2 = evaluate(rootPos);
                if (e1 != e2)
                    std::cout << "ACC MISMATCH: incremental " << e1
                              << " vs refreshed " << e2 << std::endl;
            }
        } else if (token == "d") {
            for (int r = 7; r >= 0; r--) {
                for (int f = 0; f < 8; f++) {
                    int sq = r * 8 + f;
                    char c = '.';
                    if (rootPos.board[sq] != NO_PIECE) {
                        c = "pnbrqk"[rootPos.board[sq]];
                        if (rootPos.colorOn(sq) == WHITE) c = char(toupper(c));
                    }
                    std::cout << c << ' ';
                }
                std::cout << '\n';
            }
            std::cout << "key " << std::hex << rootPos.key << std::dec << std::endl;
        }
    }
    stopSearchThread();
}

int main() {
    std::ios::sync_with_stdio(false);
    initTables();
    initZobrist();
    initCastlePerm();
    initSearch();
    tt.resize(128);
    // Deterministic default: use the exact network compiled into this binary.
    // An external network is loaded only through the explicit EvalFile option.
    resonance::loadEmbedded();
    rootPos.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    gameHist[gameHistLen++] = rootPos.key;
    uciLoop();
    return 0;
}
