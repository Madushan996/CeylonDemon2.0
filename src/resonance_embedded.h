// SPDX-License-Identifier: GPL-3.0-or-later
// CeylonDemon — Copyright (C) 2026 Madushan Dissanayake.
// Licensed under the GNU GPL v3 or later; see LICENSE.
//
// resonance_embedded.h — link the default network into the executable.
//
// The assembler drops the file in verbatim between two labels, so the 9.9 MB
// network costs nothing at compile time. `.incbin` resolves its path against
// the assembler's include path, which the build points at the network
// directory (-Wa,-I<dir>).
//
// Built without CEYLON_EMBEDDED_NET, the engine looks for ceylondemon.aa on
// disk instead. The shipping build always embeds, so a release is one file.
#pragma once

#if defined(CEYLON_EMBEDDED_NET)

    #define CEYLON_STR2(x) #x
    #define CEYLON_STR(x) CEYLON_STR2(x)

asm(".section .rodata\n"
    ".balign 64\n"
    ".globl ceylonNetStart\n"
    "ceylonNetStart:\n"
    ".incbin \"" CEYLON_STR(CEYLON_EMBEDDED_NET) "\"\n"
    ".globl ceylonNetEnd\n"
    "ceylonNetEnd:\n"
    ".balign 64\n"
    ".text\n");

extern "C" const unsigned char ceylonNetStart[];
extern "C" const unsigned char ceylonNetEnd[];

#endif
